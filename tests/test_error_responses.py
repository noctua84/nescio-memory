"""Endpoint-level tests for dependency failure responses.

These go through the real routes, so they verify the wiring in create_app() as
well as the mapping itself.
"""
import logging

import psycopg2.errors
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.api.v1 import ingest as ingest_module
from app.api.v1 import search as search_module
from app.core import errors as errors_module
from app.core.db import get_db
from app.core.errors import (
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


def _raise(exception):
    def _raiser(text):
        raise exception

    return _raiser


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
            psycopg2.errors.QueryCanceled(
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
        return [0.0] * 384

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
