"""An exhausted connection pool answers 503 "Database busy", not the catch-all's 500.

sqlalchemy.exc.TimeoutError is not an OperationalError, so without a handler of its
own it falls through to the catch-all, and a client told 500 has no reason to retry a
request that would succeed a moment later. Real exhaustion can surface wherever a
session first needs a connection, which is either in auth or in the route's work, so
both are covered, and one test exhausts a real pool rather than raising by hand.

raise_server_exceptions=False throughout: losing the handler then reads as
`assert 500 == 503`, a statement about the contract, instead of an exception out of
the test client.
"""
import logging

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.orm import sessionmaker

from app.api.v1 import ingest as ingest_module
from app.api.v1 import search as search_module
from app.core.db import get_db
from app.core.key_usage import get_key_usage_recorder
from app.main import app
from app.repositories.note import NoteRepository
from app.repositories.passage_search import PassageSearchRepository
from tests.factories import make_api_key
from tests.fakes import fake_embedding, fake_embeddings

# SQLAlchemy's own wording, which names pool internals a client has no use for.
POOL_MESSAGE = "QueuePool limit of size 5 overflow 10 reached, connection timed out"
SEARCH = ("/api/v1/search", {"q": "pelicanium", "top_k": 5})
INGEST = (
    "/api/v1/ingest",
    {"scope": "repo_a", "path_or_uri": "docs/note.md", "content": "The pelicanium isotope."},
)
BUSY_BODY = {"detail": "Database busy"}


def _raise_pool_timeout(*args, **kwargs):
    raise PoolTimeoutError(POOL_MESSAGE)


def _assert_database_busy(response) -> None:
    assert response.status_code == 503, response.text
    # The whole body: this is also what keeps the pool's message out of it.
    assert response.json() == BUSY_BODY
    assert response.headers["Retry-After"] == "30"
    assert "QueuePool" not in response.text


@pytest.fixture
def http(db_session, monkeypatch):
    monkeypatch.setattr(ingest_module, "get_embeddings", fake_embeddings)
    monkeypatch.setattr(search_module, "get_query_embedding", fake_embedding)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_key_usage_recorder] = lambda: lambda key_id: None
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("target", "method", "request_"),
    [
        pytest.param(PassageSearchRepository, "search", SEARCH, id="search"),
        pytest.param(NoteRepository, "lock_note", INGEST, id="ingest"),
    ],
)
def test_pool_timeout_during_the_routes_work_is_database_busy(
    http, db_session, monkeypatch, target, method, request_
):
    """Auth has succeeded by now, so this is the route's own checkout timing out."""
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(target, method, _raise_pool_timeout)
    path, body = request_

    response = http.post(path, json=body, headers={"X-API-Key": key})

    _assert_database_busy(response)


def test_pool_timeout_during_auth_is_database_busy(http, db_session, monkeypatch):
    """The first statement of any request is auth's, so that is where most of a
    saturated pool's requests time out, ahead of any route code."""
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(db_session, "execute", _raise_pool_timeout)

    response = http.post(SEARCH[0], json=SEARCH[1], headers={"X-API-Key": key})

    _assert_database_busy(response)


def test_a_really_exhausted_pool_answers_busy_and_recovers(engine, pg_container):
    """No stand-in for the error: this proves that what a real pool raises through a
    real session is what the handler claims, and that a retry can then succeed, which
    is what Retry-After promises."""
    small_pool = create_engine(
        pg_container.get_connection_url(), pool_size=1, max_overflow=0, pool_timeout=0.1
    )
    factory = sessionmaker(autocommit=False, autoflush=False, bind=small_pool)

    def real_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = real_get_db
    http = TestClient(app, raise_server_exceptions=False)
    headers = {"X-API-Key": "nm_no_such_key"}
    try:
        with small_pool.connect():
            exhausted = http.post(SEARCH[0], json=SEARCH[1], headers=headers)
        # The key is unknown on purpose: nothing is committed, and a 401 can only
        # come from auth having got a connection this time.
        recovered = http.post(SEARCH[0], json=SEARCH[1], headers=headers)
    finally:
        app.dependency_overrides.clear()
        small_pool.dispose()

    _assert_database_busy(exhausted)
    assert recovered.status_code == 401, recovered.text


def test_pool_exhaustion_is_logged_for_the_operator(http, db_session, monkeypatch, caplog):
    """The body says nothing specific by design, so the log is where the pool's own
    message has to survive, or saturation could not be told from anything else."""
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(PassageSearchRepository, "search", _raise_pool_timeout)

    with caplog.at_level(logging.WARNING, logger="app.core.errors"):
        http.post(SEARCH[0], json=SEARCH[1], headers={"X-API-Key": key})

    messages = [r.getMessage() for r in caplog.records if r.name == "app.core.errors"]
    assert len(messages) == 1
    assert SEARCH[0] in messages[0]
    assert POOL_MESSAGE in messages[0]


def test_the_builtin_timeout_error_is_not_mistaken_for_a_busy_pool(
    http, db_session, monkeypatch
):
    """The two classes share a name and nothing else. A socket timeout from anywhere
    else is an unanticipated failure, and blaming the database for it would be untrue."""
    key = make_api_key(db_session, "acme")

    def _raise_builtin(*args, **kwargs):
        raise TimeoutError("some other wait gave up")

    monkeypatch.setattr(PassageSearchRepository, "search", _raise_builtin)

    response = http.post(SEARCH[0], json=SEARCH[1], headers={"X-API-Key": key})

    assert response.status_code == 500, response.text
    assert "retry-after" not in response.headers
