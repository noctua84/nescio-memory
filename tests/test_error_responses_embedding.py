"""Embedding-backend failures at the /api/v2 routes: status, body, Retry-After.

Split from tests/test_error_responses.py by concern: this module pins what the
CALLER sees for each EmbeddingBackendError class; test_error_responses.py pins
what the operator's log sees; test_error_responses_database.py covers the
OperationalError and DataError handlers.

Every test here patches the raw transport, `_fetch_batch`, and leaves the real
get_query_embedding / get_embeddings -> _embed -> _validated chain in place.
Patching the route's embedder to raise would only re-test the handler and stay
green with a validation check removed; through the real chain, removing the
finiteness, float32-range or norm check turns the matching case red.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.config import settings
from app.core import embeddings as embeddings_module
from app.core.db import get_db
from app.core.errors import (
    RETRY_AFTER_SECONDS,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)
from app.main import app
from app.models.note import Note
from app.models.passage import Passage
from tests.factories import make_api_key, make_note, make_passage
from tests.fakes import fake_embeddings

BAD_RESPONSE_DETAIL = "Embedding backend returned an unexpected response"
INGEST_PAYLOAD = {
    "scope": "repo_a",
    "path_or_uri": "docs/note.md",
    "content": "The pelicanium isotope decays slowly in cool storage.",
}


@pytest.fixture
def http(db_session, monkeypatch):
    """A client on the test transaction whose only fake is the raw transport.

    Not the conftest `client` fixture: that one replaces the route modules'
    get_embeddings / get_query_embedding with fakes, which is exactly the chain
    these tests must run. Faking `_fetch_batch` instead keeps every test off the
    network by default while each test swaps in the batch it needs.

    raise_server_exceptions=False so an unhandled exception surfaces as the bare
    500 a real caller would get, i.e. as a failed assertion about the contract
    rather than as an error out of client.post().
    """
    monkeypatch.setattr(
        embeddings_module, "_fetch_batch", lambda texts, timeout: fake_embeddings(texts)
    )
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()


def _answering(make_vector):
    """A `_fetch_batch` answering every text with `make_vector()`.

    One vector per input: a short batch raises the same BadResponse, so it would
    let these tests pass without the component checks they target ever running.
    """
    return lambda texts, timeout: [make_vector() for _ in texts]


def _raising(exception):
    def fetch_batch(texts, timeout):
        raise exception

    return fetch_batch


def _with_component(index: int, value: float) -> list[float]:
    vector = [0.01] * settings.embedding_dimension
    vector[index] = value
    return vector


# Each passes every check _validated() makes before the one it is named for.
# nan and 1e39 are why this mapping exists at all: without the finiteness and
# float32-range checks pgvector rejects them with SQLSTATE 22000 / 22003, a
# DataError the handler maps to 400 -- so a broken backend would make every
# search look like a CLIENT error while 5xx alerts stayed green. Zero norm is
# #42: it passes every per-component rule, and cosine distance against it is
# NaN, which Postgres does not treat as an error, so the pre-fix response was a
# 200 with `"similarity": null`.
UNUSABLE_BATCHES = [
    pytest.param(lambda texts, timeout: [], id="empty-batch"),
    pytest.param(_answering(lambda: "not a vector"), id="not-a-list"),
    pytest.param(
        _answering(lambda: [0.01] * (settings.embedding_dimension - 1)),
        id="wrong-width",
    ),
    pytest.param(_answering(lambda: _with_component(0, float("nan"))), id="nan"),
    pytest.param(_answering(lambda: _with_component(0, 1e39)), id="float32-overflow"),
    pytest.param(
        _answering(lambda: [0.0] * settings.embedding_dimension), id="zero-norm"
    ),
]


def _search(http, key):
    return http.post(
        "/api/v2/search",
        json={"q": "pelicanium", "top_k": 5},
        headers={"X-API-Key": key},
    )


def _ingest(http, key):
    return http.post("/api/v2/ingest", json=INGEST_PAYLOAD, headers={"X-API-Key": key})


def _nothing_persisted(db_session) -> bool:
    # The route never committed, so a real request teardown discards its writes;
    # rolling the session back reproduces that. Had the route committed, the
    # rows would survive this rollback.
    db_session.rollback()
    return (
        db_session.scalar(select(func.count()).select_from(Note)) == 0
        and db_session.scalar(select(func.count()).select_from(Passage)) == 0
    )


def test_ingest_returns_503_with_retry_after_when_the_backend_is_down(
    http, db_session, monkeypatch
):
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(
        embeddings_module, "_fetch_batch", _raising(EmbeddingBackendError("refused"))
    )

    response = _ingest(http, key)

    assert response.status_code == 503
    assert response.json() == {"detail": "Embedding backend unavailable"}
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)
    # The route upserts the note row before it embeds, so there is something of
    # its own a failure could leave behind.
    assert _nothing_persisted(db_session)


def test_search_with_the_backend_down_degrades_instead_of_returning_503(
    http, db_session, monkeypatch
):
    # The v1 contract here was 503 + Retry-After. On /api/v2/search the exact
    # transient class degrades to full-text results instead; the full contract
    # (FTS-only body, the warning, subclasses that must NOT degrade) lives in
    # tests/test_search_degraded.py. This one assertion only pins that the
    # handler no longer sees this class from search.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(
        embeddings_module, "_fetch_batch", _raising(EmbeddingBackendError("refused"))
    )

    response = _search(http, key)

    assert response.status_code == 200, response.text
    assert response.json()["degraded"] is True


@pytest.mark.parametrize("fetch_batch", UNUSABLE_BATCHES)
def test_an_unusable_query_embedding_is_a_503_on_search(
    http, db_session, monkeypatch, fetch_batch
):
    # A passage is seeded so that a regression has something to return: with an
    # empty table, a broken guard would produce an empty 200 indistinguishable
    # from a correct miss.
    key = make_api_key(db_session, "acme")
    make_passage(
        db_session,
        make_note(db_session, "acme"),
        content="The pelicanium isotope decays slowly.",
    )
    monkeypatch.setattr(embeddings_module, "_fetch_batch", fetch_batch)

    response = _search(http, key)

    assert response.status_code == 503
    assert response.json() == {"detail": BAD_RESPONSE_DETAIL}
    # Retry-After present is what separates a backend fault from the DataError
    # 400 these vectors produce without the guard: that path carries none, so a
    # check of status and body alone could be satisfied by a change that got
    # the code right and the semantics wrong.
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)
    # A subclass of EmbeddingBackendError, so search must not degrade on it:
    # FTS-only results would hide a backend that answers with garbage.
    assert "results" not in response.json()
    assert "degraded" not in response.json()


@pytest.mark.parametrize("fetch_batch", UNUSABLE_BATCHES)
def test_an_unusable_passage_embedding_is_a_503_on_ingest_and_persists_nothing(
    http, db_session, monkeypatch, fetch_batch
):
    # The second route: the mapping has to be reachable from more than one call
    # site, and ingest adds the assertion search cannot make. A zero-norm row
    # stored here is what search would otherwise have to filter out at read time.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(embeddings_module, "_fetch_batch", fetch_batch)

    response = _ingest(http, key)

    assert response.status_code == 503
    assert response.json() == {"detail": BAD_RESPONSE_DETAIL}
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)
    assert _nothing_persisted(db_session)


@pytest.mark.parametrize("route", ["search", "ingest"])
def test_a_misconfigured_backend_is_500_with_no_retry_after(
    http, db_session, monkeypatch, route
):
    # A model that was never pulled will not succeed on retry, so a Retry-After
    # would tell the client something untrue. Search must not degrade on it
    # either: keyword results "for good" would hide the deployment fault.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(
        embeddings_module,
        "_fetch_batch",
        _raising(EmbeddingBackendMisconfigured("model not pulled")),
    )

    response = _search(http, key) if route == "search" else _ingest(http, key)

    assert response.status_code == 500
    assert response.json() == {"detail": "Embedding backend is misconfigured"}
    assert "Retry-After" not in response.headers
