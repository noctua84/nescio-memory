"""Database failures at the /api/v1 routes: the OperationalError and DataError handlers.

Split from tests/test_error_responses.py by concern (see that module). A real
statement cancellation by Postgres, and the guard that sets the timeout, are
pinned in tests/test_statement_timeout.py; this module pins the HTTP mapping.
"""
import logging

import psycopg.errors
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import DataError, OperationalError

from app.api.v1 import ingest as ingest_module
from app.api.v1 import search as search_module
from app.core import errors as errors_module
from app.core.db import get_db
from app.core.errors import INVALID_DATA_DETAIL, QUERY_TIMEOUT_DETAIL, RETRY_AFTER_SECONDS
from app.main import app
from app.models.note import Note
from app.models.passage import Passage
from app.repositories.passage_search import PassageSearchRepository
from tests.factories import make_api_key
from tests.fakes import fake_embedding, fake_embeddings

# The cheapest value that is a legal Python string and a legal JSON body, yet
# cannot be bound as a PostgreSQL text parameter. It reaches the DataError
# handler with nothing staged: the real driver rejects it while adapting the
# parameter, so the statement is never sent and no SQLSTATE exists.
NUL_BYTE = "\x00"
# psycopg's own wording for that rejection: an internal diagnostic that must
# never appear in a response body.
PSYCOPG_NUL_MESSAGE = "PostgreSQL text fields cannot contain NUL (0x00) bytes"

INGEST_PAYLOAD = {
    "scope": "repo_a",
    "path_or_uri": "docs/note.md",
    "content": "The pelicanium isotope decays slowly in cool storage.",
}
SEARCH_PAYLOAD = {"q": "pelicanium", "top_k": 5}


@pytest.fixture
def http(db_session, monkeypatch):
    """The conftest `client`, but returning 5xx responses instead of raising.

    With the default raise_server_exceptions=True, a DataError no handler
    claimed would raise out of client.post(); with it off, removing the handler
    shows up as `assert 500 == 400`, a statement about the contract.
    """
    monkeypatch.setattr(ingest_module, "get_embeddings", fake_embeddings)
    monkeypatch.setattr(search_module, "get_query_embedding", fake_embedding)
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def handler_log(monkeypatch, caplog):
    """Records from app.core.errors only, at WARNING and above.

    The session-scoped `engine` fixture runs Alembic, whose env.py calls
    logging.config.fileConfig(), which disables every logger that already
    existed -- app.core.errors' included. That is test wiring, so it is undone
    here rather than in app/ or alembic.ini. Filtering by name keeps the
    negative assertions about this one logger, not about whatever SQLAlchemy or
    testcontainers happened to log.
    """
    monkeypatch.setattr(errors_module.logger, "disabled", False)
    with caplog.at_level(logging.WARNING, logger="app.core.errors"):
        yield lambda: [
            r.getMessage() for r in caplog.records if r.name == "app.core.errors"
        ]


def _post(http, key, route, body):
    return http.post(f"/api/v1/{route}", json=body, headers={"X-API-Key": key})


# --- OperationalError ---------------------------------------------------------


def test_database_failure_returns_503_with_retry_after(db_session):
    class _FailingSession:
        def execute(self, *args, **kwargs):
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

        def close(self):
            pass

    # get_current_client resolves through get_db, so this raises while
    # authenticating, before the route body: the key is irrelevant, and a 401
    # would mean the handler is not reached from a dependency.
    app.dependency_overrides[get_db] = lambda: _FailingSession()
    try:
        response = TestClient(app).post(
            "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": "nm_irrelevant"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)


def test_a_canceled_search_statement_is_503_with_no_retry_after(
    http, db_session, monkeypatch, handler_log
):
    # Raised from the search path itself, after authentication, so it must be
    # mapped distinctly from the generic branch above, which promises a retry:
    # the same query over the same data times out again.
    key = make_api_key(db_session, "acme")

    def _canceled(self, *args, **kwargs):
        raise OperationalError(
            "SELECT ...",
            {},
            psycopg.errors.QueryCanceled("canceling statement due to statement timeout"),
        )

    monkeypatch.setattr(PassageSearchRepository, "search", _canceled)

    response = _post(http, key, "search", SEARCH_PAYLOAD)

    assert response.status_code == 503
    assert response.json() == {"detail": QUERY_TIMEOUT_DETAIL}
    assert "Retry-After" not in response.headers
    # The warning is the only operator-facing signal of timeout starvation.
    messages = handler_log()
    assert any("57014" in m and "/api/v1/search" in m for m in messages), messages


# --- DataError ----------------------------------------------------------------


@pytest.mark.parametrize("field", ["scope", "path_or_uri", "content", "status"])
def test_a_nul_byte_in_an_ingest_field_is_a_400_with_no_retry_after(
    http, db_session, field
):
    # scope and path_or_uri fail in lock_note's SELECT, the rest in the note
    # upsert; both moments must land on the same response.
    key = make_api_key(db_session, "acme")
    body = {**INGEST_PAYLOAD, "status": "draft"}
    body[field] = body[field] + NUL_BYTE

    response = _post(http, key, "ingest", body)

    assert response.status_code == 400
    assert response.json() == {"detail": INVALID_DATA_DETAIL}
    # The same bytes are rejected on every attempt; this is what distinguishes
    # the DataError mapping from OperationalError's.
    assert "Retry-After" not in response.headers


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({**SEARCH_PAYLOAD, "q": f"pelican{NUL_BYTE}ium"}, id="q"),
        pytest.param({**SEARCH_PAYLOAD, "repo_filter": f"repo_a{NUL_BYTE}"}, id="repo_filter"),
    ],
)
def test_a_nul_byte_in_a_search_field_is_a_400(http, db_session, body):
    # Unlike the removed learnings API's vector-only search, `q` is now a bound SQL parameter too:
    # the full-text arm passes it to websearch_to_tsquery.
    key = make_api_key(db_session, "acme")

    response = _post(http, key, "search", body)

    assert response.status_code == 400
    assert response.json() == {"detail": INVALID_DATA_DETAIL}
    assert "Retry-After" not in response.headers


def test_the_drivers_nul_byte_message_does_not_reach_the_client(http, db_session):
    # `detail` must come from the class constant; a handler that reached for
    # str(exc) would pass the status assertions above and still hand the caller
    # psycopg's internals and the SQL SQLAlchemy attaches.
    key = make_api_key(db_session, "acme")

    response = _post(http, key, "ingest", {**INGEST_PAYLOAD, "scope": f"repo_a{NUL_BYTE}"})

    assert response.status_code == 400
    assert PSYCOPG_NUL_MESSAGE not in response.text
    assert "psycopg" not in response.text.lower()
    assert "SELECT" not in response.text
    assert response.json() == {"detail": INVALID_DATA_DETAIL}


def test_a_client_side_rejection_is_logged_with_the_path_and_an_unknown_sqlstate(
    http, db_session, handler_log
):
    # psycopg rejects the NUL while adapting the parameter, so Postgres never
    # assigns a SQLSTATE: the handler must print "unknown" rather than crash or
    # print "None", which is what tells this apart from a server-side 22xxx.
    key = make_api_key(db_session, "acme")

    response = _post(http, key, "ingest", {**INGEST_PAYLOAD, "scope": f"repo_a{NUL_BYTE}"})

    assert response.status_code == 400
    messages = handler_log()
    assert any("/api/v1/ingest" in m for m in messages), messages
    assert any("SQLSTATE unknown" in m for m in messages), messages
    # The log may carry internal detail, but must not be the response's source.
    assert all(INVALID_DATA_DETAIL not in m for m in messages)


def test_a_server_side_22xxx_is_also_a_400_and_its_sqlstate_is_logged(
    http, db_session, monkeypatch, handler_log
):
    # A DataError Postgres itself raised (numeric_value_out_of_range), the shape
    # the rest of class 22xxx takes. Staged at the repository because no
    # caller-reachable input overflows a column: top_k is clamped by Pydantic
    # before it is bound. Two things the NUL tests cannot show: a genuine
    # SQLSTATE still maps to 400, and the log prints it rather than "unknown",
    # which keeps an internal-SQL bug visible instead of blamed on the client.
    key = make_api_key(db_session, "acme")

    def _overflowed(self, *args, **kwargs):
        raise DataError(
            "SELECT ...", {}, psycopg.errors.NumericValueOutOfRange("value out of range")
        )

    monkeypatch.setattr(PassageSearchRepository, "search", _overflowed)

    response = _post(http, key, "search", SEARCH_PAYLOAD)

    assert response.status_code == 400
    assert response.json() == {"detail": INVALID_DATA_DETAIL}
    assert "Retry-After" not in response.headers
    messages = handler_log()
    assert any("22003" in m and "/api/v1/search" in m for m in messages), messages
    assert all("unknown" not in m for m in messages), messages


def test_a_rejected_reingest_leaves_the_previous_note_and_passages_intact(
    http, db_session
):
    # Ported from the learnings API's commit-time rejection test, which test_ingest*.py
    # does not cover: there, only an EMBEDDING failure is shown to leave the
    # previous passages intact. Here the failure is a DataError after
    # lock_note has taken the row lock on the existing note, i.e. inside the
    # same transaction that would go on to replace its passages. A route that
    # committed anything before the failure would answer 400 and lose data.
    key = make_api_key(db_session, "acme")
    first = _post(http, key, "ingest", INGEST_PAYLOAD).json()

    def snapshot():
        db_session.expire_all()
        note = db_session.execute(select(Note).where(Note.id == first["note_id"])).scalar_one()
        passages = db_session.execute(
            select(Passage.id, Passage.content).where(Passage.note_id == note.id)
        ).all()
        return note.body, note.content_hash, sorted(passages)

    before = snapshot()

    response = _post(
        http,
        key,
        "ingest",
        {**INGEST_PAYLOAD, "content": "A replacement body with a NUL" + NUL_BYTE},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": INVALID_DATA_DETAIL}
    # No commit happened on the failing request, so a real teardown discards
    # its writes; rolling back reproduces that.
    db_session.rollback()
    assert snapshot() == before
