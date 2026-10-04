"""Endpoint-level tests for dependency failure responses.

These go through the real routes, so they verify the wiring in create_app() as
well as the mapping itself.
"""
import logging
from contextlib import contextmanager

import psycopg.errors
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import DataError, OperationalError
from sqlalchemy.orm import Session

from app.api.v1 import ingest as ingest_module
from app.api.v1 import search as search_module
from app.config import settings
from app.core import embeddings as embeddings_module
from app.core import errors as errors_module
from app.core.db import get_db
from app.core.errors import (
    INVALID_DATA_DETAIL,
    QUERY_TIMEOUT_DETAIL,
    RETRY_AFTER_SECONDS,
    EmbeddingBackendBadResponse,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)
from app.main import app
from app.models.learning import Learning
from app.repositories.learning import LearningRepository
from tests.factories import make_api_key, make_learning
from tests.fakes import fake_embedding

LONG_ENOUGH = (
    "This content comfortably exceeds the fifty character minimum that the "
    "ingest endpoint applies to each chunk."
)
INGEST_PAYLOAD = {
    "repo_name": "repo_a",
    "file_path": "docs/note.md",
    "content": LONG_ENOUGH,
}
SEARCH_PAYLOAD = {"query": "anything", "top_k": 5}
ORIGINAL_CONTENT = "the original content, which must survive a failed re-ingest"

# A NUL byte is the cheapest value that is a perfectly legal Python string and a
# perfectly legal HTTP body, yet cannot be represented in a PostgreSQL text
# column. It is therefore the one input that reaches the DataError handler
# without any monkeypatching at all -- the failure is produced by the real
# driver on the real database, not staged.
NUL_BYTE = "\x00"

# psycopg's own wording for that rejection. It is an internal diagnostic and the
# assertion below is that it never appears in a response body.
PSYCOPG_NUL_MESSAGE = "PostgreSQL text fields cannot contain NUL (0x00) bytes"

BAD_RESPONSE_DETAIL = "Embedding backend returned an unexpected response"


def _vector_with_a_non_finite_component() -> list[float]:
    """A vector that is the right width and the right type, and still unusable.

    Every component is a genuine float and there are exactly
    EMBEDDING_DIMENSION of them, so the list, length and component-type checks
    in _validated() all pass and only its finiteness check can reject this.
    That is the point: a NaN is not a malformed response in any way the older
    checks could see.

    Built in Python rather than decoded from a raw JSON body because this
    module's subject is the endpoint contract. That the bare `NaN` / `Infinity`
    literals really arrive off the wire as these values is pinned separately, in
    tests/test_embeddings.py.
    """
    vector = [0.01] * settings.embedding_dimension
    vector[0] = float("nan")
    return vector


def _raise(exception):
    def _raiser(text):
        raise exception

    return _raiser


def _handler_log_messages(caplog):
    """The formatted messages this module's handler logger emitted, and no others.

    caplog captures every logger in the process, so iterating caplog.records
    directly makes an assertion about the whole run rather than about
    app.core.errors. The `all(... not in message ...)` assertions are where that
    actually bites: any unrelated library warning that happens to contain the
    substring -- "unknown" is an ordinary English word, and SQLAlchemy, httpx and
    testcontainers all log during these requests -- would fail the test for a
    reason with nothing to do with the handler. Filtering by logger name keeps
    the negative assertions about the one logger whose output is the contract.

    getMessage() rather than caplog.text is also deliberate: the handler now
    passes exc_info, so caplog.text carries the driver's traceback (including
    psycopg's NUL wording). That belongs in the log and is exactly what the
    exc_info change is for, but it would make a naive substring search over
    caplog.text match things this test is not asserting about.
    """
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "app.core.errors"
    ]


@contextmanager
def _client_that_reports_server_errors(db_session):
    """A TestClient wired to the test transaction that returns 5xx responses.

    Two departures from the `client` fixture, both deliberate.

    `raise_server_exceptions=False` is the important one, and no other test in
    this suite needs it. With the default (True), Starlette's TestClient
    re-raises any exception that no handler claimed instead of producing a
    response. For the DataError tests that would be actively misleading: if the
    handler were removed or stopped matching, the tests would fail by raising
    sqlalchemy.exc.DataError out of client.post() rather than by observing the
    bare 500 a real caller would get. With the flag off, the test asserts on the
    same bytes an HTTP client would see in production either way, so a removal of
    the handler shows up as `assert 500 == 400` -- a statement about the contract
    -- rather than as an error in the test harness.

    Second, the override is installed here rather than by the fixture only
    because the fixture hands back an already-constructed TestClient and the flag
    has to be passed to the constructor. Clearing in `finally` matches
    test_database_failure_returns_503: a leaked override carries this test's
    session into the next one and fails somewhere misleading.
    """
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()


def test_ingest_returns_503_when_the_embedding_backend_is_down(
    client, db_session, monkeypatch
):
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(
        ingest_module, "get_embedding", _raise(EmbeddingBackendError("refused"))
    )

    response = client.post(
        "/api/v1/ingest", data=INGEST_PAYLOAD, headers={"X-API-Key": key}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Embedding backend unavailable"}
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)

    # Nothing may be persisted. delete_by_file() runs before the first insert, so
    # a failure that left the transaction committed would lose data rather than
    # merely fail.
    from sqlalchemy import func, select

    from app.models.learning import Learning

    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Learning)) == 0


def test_search_returns_503_when_the_embedding_backend_is_down(
    client, db_session, monkeypatch
):
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(
        search_module, "get_embedding", _raise(EmbeddingBackendError("refused"))
    )

    response = client.post(
        "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Embedding backend unavailable"}
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)


def test_an_unreadable_backend_response_is_reported_distinctly(
    client, db_session, monkeypatch
):
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(
        search_module,
        "get_embedding",
        _raise(EmbeddingBackendBadResponse("no embedding key")),
    )

    response = client.post(
        "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Embedding backend returned an unexpected response"
    }


def test_a_misconfigured_backend_is_500_with_no_retry_after(
    client, db_session, monkeypatch
):
    # A missing extra will never succeed on retry, so advertising Retry-After
    # would tell the client something untrue.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(
        search_module,
        "get_embedding",
        _raise(EmbeddingBackendMisconfigured("extra not installed")),
    )

    response = client.post(
        "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Embedding backend is misconfigured"}
    assert "Retry-After" not in response.headers


def test_internal_exception_messages_do_not_reach_the_client(
    client, db_session, monkeypatch
):
    # The exception carries an internal message for logs; the response must show
    # only the class's fixed detail.
    key = make_api_key(db_session, "acme")
    secret = "http://internal-ollama.corp:11434 refused the connection"
    monkeypatch.setattr(
        search_module, "get_embedding", _raise(EmbeddingBackendError(secret))
    )

    response = client.post(
        "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
    )

    assert secret not in response.text
    assert response.json() == {"detail": "Embedding backend unavailable"}


def test_database_failure_returns_503(db_session, monkeypatch):
    # Built without the `client` fixture, because this test needs a session that
    # fails rather than the working one that fixture installs.
    from fastapi.testclient import TestClient

    from tests.fakes import fake_embedding

    monkeypatch.setattr(search_module, "get_embedding", fake_embedding)

    class _FailingSession:
        def execute(self, *args, **kwargs):
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

        def close(self):
            pass

    # The key is irrelevant: get_current_client resolves through get_db, so the
    # OperationalError is raised while authenticating, before the endpoint body.
    # A 401 here would mean the handler is not reached from a dependency.
    app.dependency_overrides[get_db] = lambda: _FailingSession()
    try:
        response = TestClient(app).post(
            "/api/v1/search",
            json=SEARCH_PAYLOAD,
            headers={"X-API-Key": "nm_irrelevant"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)


def test_a_search_that_exceeds_its_statement_timeout_returns_503_with_no_retry_after(
    client, db_session, monkeypatch, caplog
):
    # This must arise from the search path itself (not a fake session that
    # fails on every execute, which would raise during auth and never reach
    # LearningRepository.search) -- the handler must map it distinctly from
    # the generic OperationalError branch above, which still promises a retry.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(search_module, "get_embedding", fake_embedding)

    def _timed_out(self, *args, **kwargs):
        raise OperationalError(
            "SELECT ...",
            {},
            psycopg.errors.QueryCanceled(
                "canceling statement due to statement timeout"
            ),
        )

    monkeypatch.setattr(LearningRepository, "search", _timed_out)

    # The session-scoped `engine` fixture runs Alembic migrations, and
    # alembic/env.py calls logging.config.fileConfig(), which (default
    # disable_existing_loggers=True) disables every logger that already
    # existed and isn't named in alembic.ini -- including this module's
    # logger, created at import time when conftest imports app.main. That is
    # an artifact of test wiring, not of the application, so it is undone
    # here rather than by changing app/ or alembic.ini.
    monkeypatch.setattr(errors_module.logger, "disabled", False)

    with caplog.at_level(logging.WARNING, logger="app.core.errors"):
        response = client.post(
            "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
        )

    assert response.status_code == 503
    assert response.json() == {"detail": QUERY_TIMEOUT_DETAIL}
    assert "Retry-After" not in response.headers
    # The warning is the only operator-facing signal of HNSW/timeout
    # starvation, so it must actually be emitted, mention the sqlstate, and
    # name the request path it happened on.
    messages = [record.getMessage() for record in caplog.records]
    assert any("57014" in message for message in messages)
    assert any("/api/v1/search" in message for message in messages)


@pytest.mark.parametrize("field", ["repo_name", "file_path", "content"])
def test_a_nul_byte_in_an_ingest_field_is_a_400_with_no_retry_after(
    db_session, monkeypatch, field
):
    # The whole DataError mapping, end to end, with nothing staged: a real form
    # post, the real route, the real psycopg driver. Before the handler existed
    # this returned 500 with a text/plain "Internal Server Error" body.
    #
    # Parametrized rather than written three times because the three fields are
    # not three copies of one case -- they fail at two different moments in the
    # request, and both must land on the same response:
    #   repo_name / file_path  -> bound into the DELETE that delete_by_file()
    #                             issues before anything is embedded, so the
    #                             error is raised early in the endpoint body;
    #   content                -> survives chunking and only fails when the
    #                             INSERTs are flushed at db.commit(), i.e. after
    #                             a successful DELETE and a successful embedding.
    # A handler that only covered one of those moments would still look correct
    # against a single-field test.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(ingest_module, "get_embedding", fake_embedding)

    payload = dict(INGEST_PAYLOAD)
    payload[field] = payload[field] + NUL_BYTE

    with _client_that_reports_server_errors(db_session) as http:
        response = http.post(
            "/api/v1/ingest", data=payload, headers={"X-API-Key": key}
        )

    assert response.status_code == 400
    assert response.json() == {"detail": INVALID_DATA_DETAIL}
    # No Retry-After, unlike every 503 above. The same bytes will be rejected on
    # every attempt, so a retry hint here would be a promise the server cannot
    # keep. This assertion is the one that distinguishes the DataError mapping
    # from OperationalError's, which is otherwise the nearest neighbour.
    assert "Retry-After" not in response.headers


def test_a_nul_byte_in_the_search_repo_filter_is_a_400(db_session, monkeypatch):
    # The handler must be reachable from more than one route, so this is the
    # second one. repo_filter -- not `query` -- is the field to use, and that was
    # checked rather than assumed: search hands `query` to get_embedding and
    # sends the resulting *vector* to Postgres, so the raw query text is never a
    # SQL parameter and a NUL in it is simply searched for and not found (see the
    # test immediately below, which pins that). repo_filter is the one search
    # input that becomes a bound text parameter, via
    # `Learning.repo_name == repo_filter`, so it is the field that can actually
    # reach the driver's parameter adaptation.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(search_module, "get_embedding", fake_embedding)

    with _client_that_reports_server_errors(db_session) as http:
        response = http.post(
            "/api/v1/search",
            json={**SEARCH_PAYLOAD, "repo_filter": f"repo_a{NUL_BYTE}"},
            headers={"X-API-Key": key},
        )

    assert response.status_code == 400
    assert response.json() == {"detail": INVALID_DATA_DETAIL}
    assert "Retry-After" not in response.headers


def test_a_nul_byte_in_the_search_query_text_never_reaches_the_database(
    db_session, monkeypatch
):
    # The premise the test above rests on, pinned so it cannot quietly stop being
    # true. Search is vector search: the query string's only destination is the
    # embedding backend, and what goes to Postgres is a list of floats. So an
    # unrepresentable byte in `query` is not a database concern at all, and the
    # search runs to completion against the real table rather than being mapped
    # to 400. If someone later adds a lexical/trigram fallback that binds the raw
    # query text, this test turns red, which is the correct moment to decide
    # whether that path should be mapped too.
    #
    # A row is seeded deliberately. Asserting an empty result list would be
    # vacuous -- an empty table returns [] whether the query ran or not, so the
    # assertion would prove only "not a 400". With a row present, a 200 carrying
    # that row's content is positive evidence that the statement reached Postgres
    # and came back, which is the claim this test is actually making.
    key = make_api_key(db_session, "acme")
    make_learning(db_session, "acme", content=ORIGINAL_CONTENT)
    monkeypatch.setattr(search_module, "get_embedding", fake_embedding)

    with _client_that_reports_server_errors(db_session) as http:
        response = http.post(
            "/api/v1/search",
            json={**SEARCH_PAYLOAD, "query": f"hello{NUL_BYTE}world"},
            headers={"X-API-Key": key},
        )

    assert response.status_code == 200
    contents = [result["content"] for result in response.json()["results"]]
    assert contents == [ORIGINAL_CONTENT]


def test_the_drivers_nul_byte_message_does_not_reach_the_client(
    db_session, monkeypatch
):
    # The sibling of test_internal_exception_messages_do_not_reach_the_client,
    # for the one family whose exception text is produced by the driver rather
    # than by this codebase. `detail` must come from the class constant; a
    # handler that reached for str(exc) would pass the status-code assertions
    # above and still hand the caller psycopg's internals.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(ingest_module, "get_embedding", fake_embedding)

    with _client_that_reports_server_errors(db_session) as http:
        response = http.post(
            "/api/v1/ingest",
            data={**INGEST_PAYLOAD, "repo_name": f"repo_a{NUL_BYTE}"},
            headers={"X-API-Key": key},
        )

    assert response.status_code == 400
    assert PSYCOPG_NUL_MESSAGE not in response.text
    # Not just the exact sentence: no fragment of the driver's vocabulary, and no
    # trace of the SQL statement SQLAlchemy attaches to the exception.
    assert "psycopg" not in response.text.lower()
    assert "DELETE" not in response.text
    assert response.json() == {"detail": INVALID_DATA_DETAIL}


def test_a_client_side_rejection_is_logged_with_the_path_and_an_unknown_sqlstate(
    db_session, monkeypatch, caplog
):
    # The warning is the only operator-facing trace of a 400 from this family, so
    # it has to actually be emitted and has to name where it happened.
    #
    # The SQLSTATE assertion records a real property of this path rather than a
    # nicety: psycopg rejects the NUL byte itself, while adapting the parameter,
    # so the statement is never sent and Postgres never assigns a SQLSTATE --
    # exc.orig.sqlstate is None here. The handler must therefore degrade to
    # "unknown" instead of crashing or printing "None", and an operator reading
    # the log must be able to tell this case apart from a server-side 22xxx (the
    # next test), which does carry a code.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(ingest_module, "get_embedding", fake_embedding)

    # Same alembic/env.py fileConfig() hazard as the statement-timeout test
    # above: the session-scoped `engine` fixture runs the migrations, which
    # disables every logger that already existed, app.core.errors' included.
    # Undone here rather than in app/ or alembic.ini, because it is test wiring.
    monkeypatch.setattr(errors_module.logger, "disabled", False)

    with caplog.at_level(logging.WARNING, logger="app.core.errors"):
        with _client_that_reports_server_errors(db_session) as http:
            response = http.post(
                "/api/v1/ingest",
                data={**INGEST_PAYLOAD, "repo_name": f"repo_a{NUL_BYTE}"},
                headers={"X-API-Key": key},
            )

    assert response.status_code == 400
    messages = _handler_log_messages(caplog)
    assert any("/api/v1/ingest" in message for message in messages), messages
    assert any("SQLSTATE unknown" in message for message in messages), messages
    # The log is allowed internal detail, but it must not be the response's
    # source: the two must not have converged on the same string.
    assert all(INVALID_DATA_DETAIL not in message for message in messages)


def test_a_server_side_22xxx_is_also_a_400_and_its_sqlstate_is_logged(
    db_session, monkeypatch, caplog
):
    # The other half of the family. Everything above exercises the client-side
    # adaptation failure, where no SQLSTATE exists; this is a DataError that
    # PostgreSQL itself raised -- numeric_value_out_of_range, SQLSTATE 22003 --
    # which is the shape SQLAlchemy produces for the rest of class 22xxx
    # (invalid text representation, division by zero, datetime field overflow).
    #
    # It is injected at the repository rather than provoked with real SQL because
    # no caller-reachable input to this service overflows a column: the schema's
    # text columns are unbounded and top_k is clamped to 1..50 by Pydantic before
    # it is ever bound. Staging it is the only way to assert the branch, and what
    # is being asserted is the handler's behaviour, not SQLAlchemy's. The
    # precedent is test_a_search_that_exceeds_its_statement_timeout_..., which
    # patches the same method for the same reason.
    #
    # Two things must hold that the NUL tests cannot show: a DataError carrying a
    # genuine SQLSTATE still maps to 400 (the handler does not key off the
    # absence of one), and the warning prints that code rather than "unknown",
    # which is what lets an operator see an internal-SQL bug mis-reported as a
    # client error instead of it hiding behind a generic 400 forever.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(search_module, "get_embedding", fake_embedding)

    def _overflowed(self, *args, **kwargs):
        raise DataError(
            "SELECT ...",
            {},
            psycopg.errors.NumericValueOutOfRange("value out of range"),
        )

    monkeypatch.setattr(LearningRepository, "search", _overflowed)
    monkeypatch.setattr(errors_module.logger, "disabled", False)

    with caplog.at_level(logging.WARNING, logger="app.core.errors"):
        with _client_that_reports_server_errors(db_session) as http:
            response = http.post(
                "/api/v1/search",
                json=SEARCH_PAYLOAD,
                headers={"X-API-Key": key},
            )

    assert response.status_code == 400
    assert response.json() == {"detail": INVALID_DATA_DETAIL}
    assert "Retry-After" not in response.headers

    messages = _handler_log_messages(caplog)
    assert any("22003" in message for message in messages), messages
    assert any("/api/v1/search" in message for message in messages), messages
    assert all("unknown" not in message for message in messages), messages


def test_health_is_unaffected(client):
    # /health must keep contacting nothing, so a dependency outage cannot turn a
    # liveness probe into a restart.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_a_failure_partway_through_a_file_leaves_earlier_rows_intact(
    connection, monkeypatch
):
    """Ingest must be all-or-nothing.

    delete_by_file() runs before the first insert, so a failure midway could
    otherwise leave the caller's previous chunks deleted and the new ones absent.

    This test deliberately does NOT use the `client` fixture. That fixture
    overrides get_db with `lambda: db_session`, a plain function with no teardown,
    so the `finally: db.close()` that performs the rollback in production never
    runs and the rollback cannot be observed. The override below is generator
    shaped, like the real get_db.
    """
    from fastapi.testclient import TestClient

    # Setup lives in its own session and is committed, so releasing the endpoint's
    # savepoint later cannot take the setup row with it.
    setup = Session(bind=connection, join_transaction_mode="create_savepoint")
    key = make_api_key(setup, "acme")
    make_learning(
        setup,
        "acme",
        repo_name="repo_a",
        file_path="docs/note.md",
        content=ORIGINAL_CONTENT,
    )
    setup.commit()

    calls = {"n": 0}

    def fail_on_the_second_chunk(text):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise EmbeddingBackendError("backend died mid-file")
        return [0.0] * settings.embedding_dimension

    monkeypatch.setattr(ingest_module, "get_embedding", fail_on_the_second_chunk)

    def production_shaped_get_db():
        request_session = Session(
            bind=connection, join_transaction_mode="create_savepoint"
        )
        try:
            yield request_session
        finally:
            # Closing releases this session's savepoint. Note the nuance: in
            # production the rollback is guaranteed by db.commit() never being
            # reached, and close() serves to release the transaction and its locks
            # promptly. Removing this line does make THIS test fail, because the
            # assertions read through the same connection the savepoint sits on.
            request_session.close()

    app.dependency_overrides[get_db] = production_shaped_get_db
    try:
        # 2000 characters is 3 chunks at chunk_size 1000 with overlap 200, so the
        # failure lands after at least one successful embedding.
        response = TestClient(app).post(
            "/api/v1/ingest",
            data={**INGEST_PAYLOAD, "content": "x" * 2000},
            headers={"X-API-Key": key},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert calls["n"] == 2, "expected the second chunk to be the one that failed"

    check = Session(bind=connection, join_transaction_mode="create_savepoint")
    surviving = check.scalars(
        select(Learning).where(Learning.client_name == "acme")
    ).all()
    assert len(surviving) == 1
    assert surviving[0].content == ORIGINAL_CONTENT
    assert check.scalar(select(func.count()).select_from(Learning)) == 1


def test_a_commit_time_rejection_leaves_the_existing_chunks_intact(
    connection, monkeypatch
):
    """A DataError raised by the commit itself must still be all-or-nothing.

    The sibling above, test_a_failure_partway_through_a_file_leaves_earlier_rows_
    intact, does NOT cover this. Its failure arrives from get_embedding, i.e.
    inside the loop and strictly before db.commit(). A NUL byte in `content`
    fails at a genuinely different moment: the DELETE has already been issued
    and every embedding has already succeeded, and the rejection happens only
    when the INSERTs are flushed at the commit. That is the worst moment for it
    to happen -- delete_by_file() has notionally removed the caller's previous
    chunks for this file, so a commit that half-applied would answer 400 and
    destroy data in the same breath.

    The `client` fixture and this module's _client_that_reports_server_errors
    helper both override get_db with `lambda: db_session` -- a plain function,
    not a generator, so the `finally: db.close()` that performs the rollback in
    production never runs and the rollback cannot be observed from the
    assertions. The override below is generator shaped, like the real get_db,
    which is the same reason the sibling test builds its own.

    raise_server_exceptions=False is still required on top of that, for the
    reason given on the helper: without it a missing handler would raise out of
    client.post() instead of letting this test see the response and then check
    the rows.
    """
    # Setup lives in its own session and is committed, so releasing the
    # endpoint's savepoint later cannot take the setup row with it.
    setup = Session(bind=connection, join_transaction_mode="create_savepoint")
    key = make_api_key(setup, "acme")
    make_learning(
        setup,
        "acme",
        repo_name=INGEST_PAYLOAD["repo_name"],
        file_path=INGEST_PAYLOAD["file_path"],
        content=ORIGINAL_CONTENT,
    )
    setup.commit()

    # Embedding must succeed for all chunks: the point is to reach the commit
    # with real rows pending, not to fail earlier for an unrelated reason.
    monkeypatch.setattr(ingest_module, "get_embedding", fake_embedding)

    def production_shaped_get_db():
        request_session = Session(
            bind=connection, join_transaction_mode="create_savepoint"
        )
        try:
            yield request_session
        finally:
            request_session.close()

    app.dependency_overrides[get_db] = production_shaped_get_db
    try:
        # Same repo_name and file_path as the seeded row, so delete_by_file()
        # genuinely targets it. The NUL rides on the content, which survives
        # chunking untouched and is only rejected when the INSERT is flushed.
        response = TestClient(app, raise_server_exceptions=False).post(
            "/api/v1/ingest",
            data={**INGEST_PAYLOAD, "content": LONG_ENOUGH + NUL_BYTE},
            headers={"X-API-Key": key},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json() == {"detail": INVALID_DATA_DETAIL}

    check = Session(bind=connection, join_transaction_mode="create_savepoint")
    surviving = check.scalars(
        select(Learning).where(Learning.client_name == "acme")
    ).all()
    # The pre-existing chunk is still there, with its original text: the DELETE
    # was rolled back with the failed INSERT rather than surviving it.
    assert len(surviving) == 1
    assert surviving[0].content == ORIGINAL_CONTENT
    # And nothing from the rejected request was partially written.
    assert check.scalar(select(func.count()).select_from(Learning)) == 1


def test_a_degenerate_embedding_is_a_503_on_search_and_not_a_400(
    db_session, monkeypatch
):
    # The alerting contract, pinned. This is the one test in the suite that
    # catches a revert of the finiteness check, and it is built to discriminate.
    #
    # _fetch is what gets patched, not get_embedding. search_module.get_embedding
    # is left as the real function so the real get_embedding -> _validated chain
    # runs and raises on its own. Patching get_embedding to raise
    # EmbeddingBackendBadResponse directly would only re-test the handler mapping
    # -- which test_an_unreadable_backend_response_is_reported_distinctly already
    # covers -- and would still pass with the finiteness check removed, making it
    # worthless as a guard.
    #
    # Without that check the NaN reaches pgvector, which rejects it with SQLSTATE
    # 22000 on the distance comparison, producing a DataError that the handler
    # above maps to 400. So a wholly broken embedding backend would render every
    # search a *client* error: 5xx alerts, error-rate dashboards and SLOs all
    # stay green while the service returns nothing usable. 503 is the honest
    # answer, because the backend answered with something unreadable and that is
    # not the caller's fault.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(
        embeddings_module, "_fetch", lambda text: _vector_with_a_non_finite_component()
    )

    with _client_that_reports_server_errors(db_session) as http:
        response = http.post(
            "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
        )

    assert response.status_code == 503
    assert response.json() == {"detail": BAD_RESPONSE_DETAIL}
    # Retry-After *present* is the assertion that distinguishes this from the
    # inversion it guards against. The DataError 400 this path produces without
    # the fix carries no Retry-After (see the NUL-byte tests above), so a test
    # that checked only the status and body would still be satisfied by a future
    # handler change that got the code right and the semantics wrong. The header
    # is also the substantive difference for a caller: a transient backend fault
    # is worth retrying, and a rejected request never is.
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)


def test_a_degenerate_embedding_is_a_503_on_ingest_and_persists_nothing(
    db_session, monkeypatch
):
    # The second route, for the same reason the NUL-byte tests cover two: the
    # mapping has to be reachable from more than one call site. Ingest also adds
    # the persistence assertion search cannot make -- delete_by_file() runs
    # before the first embedding, so a failure that left the transaction
    # committed would lose the file's existing chunks rather than merely fail.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(
        embeddings_module, "_fetch", lambda text: _vector_with_a_non_finite_component()
    )

    with _client_that_reports_server_errors(db_session) as http:
        response = http.post(
            "/api/v1/ingest", data=INGEST_PAYLOAD, headers={"X-API-Key": key}
        )

    assert response.status_code == 503
    assert response.json() == {"detail": BAD_RESPONSE_DETAIL}
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)

    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Learning)) == 0


def _vector_with_a_float32_overflowing_component() -> list[float]:
    """A vector of genuine finite floats that pgvector still cannot store.

    Sibling of `_vector_with_a_non_finite_component()`, and the harder case of
    the two. Every component is finite -- math.isfinite(1e39) is True -- the
    width is exactly EMBEDDING_DIMENSION, and 1e39 is an ordinary JSON number
    that needs no non-standard literal to reach us. So the list, length,
    component-type *and* finiteness checks in _validated() all pass, and only
    its float32-range check can reject this.

    1e39 rather than something larger because it is barely over the bound: the
    point is that the value looks entirely unremarkable until it meets a float4
    column.
    """
    vector = [0.01] * settings.embedding_dimension
    vector[0] = 1e39
    return vector


def test_a_float32_overflowing_embedding_is_a_503_on_search_and_not_a_400(
    db_session, monkeypatch
):
    # The same alerting inversion as the NaN tests above, reached by a value no
    # JSON extension is needed to express -- which makes it the likelier of the
    # two to arrive from a real backend.
    #
    # pgvector's `vector` is float4. Without the range check in _validated(),
    # 1e39 reaches the database and is rejected with SQLSTATE 22003 ("1e+39" is
    # out of range for type vector) on the distance comparison. That surfaces as
    # sqlalchemy.exc.DataError, which the handler maps to 400 with no
    # Retry-After -- so a wholly broken embedding backend would make every
    # search look like a client error while 5xx alerts, error-rate dashboards
    # and SLOs all stayed green. That mapping is what this test pins against.
    #
    # _fetch is patched rather than get_embedding, and search_module.get_embedding
    # is left as the real function, for the reason spelled out on the NaN test:
    # patching get_embedding to raise would only re-test the handler and would
    # stay green with the range check removed.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(
        embeddings_module,
        "_fetch",
        lambda text: _vector_with_a_float32_overflowing_component(),
    )

    with _client_that_reports_server_errors(db_session) as http:
        response = http.post(
            "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
        )

    assert response.status_code == 503
    assert response.json() == {"detail": BAD_RESPONSE_DETAIL}
    # Retry-After present is what separates this from the 400 it guards against:
    # the DataError path carries none, so a test checking only status and body
    # could still be satisfied by a change that got the code right and the
    # semantics wrong. One route is enough here -- the two NaN tests above
    # already prove the handler is reachable from both ingest and search.
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)
