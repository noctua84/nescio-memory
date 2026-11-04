"""Endpoint-level tests for dependency failure responses: the embedding handler's log.

These go through the real /api/v1 routes, so they verify the wiring in
create_app() as well as the mapping itself. Split by concern:

- this module: what an embedding-backend failure logs for the operator, and that
  none of it reaches the caller;
- tests/test_error_responses_embedding.py: status, body and Retry-After per
  EmbeddingBackendError class, through the real validation chain;
- tests/test_error_responses_database.py: the OperationalError (including
  SQLSTATE 57014) and DataError handlers.

The client is shown nothing but each class's fixed detail, so the diagnostics
raised in app.core.embeddings (which component was NaN, what the backend sent)
survive only in the handler's log line, via exc_info.
"""
import logging
import traceback

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core import embeddings as embeddings_module
from app.core.db import get_db
from app.core.errors import (
    RETRY_AFTER_SECONDS,
    EmbeddingBackendBadResponse,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)
from app.core.key_usage import get_key_usage_recorder
from app.main import app
from tests.factories import make_api_key
from tests.fakes import fake_embeddings

BAD_RESPONSE_DETAIL = "Embedding backend returned an unexpected response"
INGEST_PAYLOAD = {
    "scope": "repo_a",
    "path_or_uri": "docs/note.md",
    "content": "The pelicanium isotope decays slowly in cool storage.",
}
# Internal detail of the kind app.core.embeddings composes: fine in a log,
# never in a response.
INTERNAL = (
    "embedding component was nan at index 7 of 1024 from "
    "http://internal-ollama.example:11434"
)


@pytest.fixture
def http(db_session, monkeypatch):
    """A client whose only fake is the raw transport, `_fetch_batch`.

    Leaves the route modules' real embedders in place (unlike the conftest
    `client`), so a failure staged on `_fetch_batch` travels the same chain a
    real backend fault would. raise_server_exceptions=False: an unhandled
    exception must surface as the 500 a caller would see, not as a test error.
    """
    monkeypatch.setattr(
        embeddings_module, "_fetch_batch", lambda texts, timeout: fake_embeddings(texts)
    )
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_key_usage_recorder] = lambda: lambda key_id: None
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def handler_records(caplog):
    """Records from app.core.errors only.

    Filtering by name keeps the assertions about this logger rather than about
    the whole process.
    """
    with caplog.at_level(logging.WARNING, logger="app.core.errors"):
        yield lambda: [r for r in caplog.records if r.name == "app.core.errors"]


def _fail_fetch_with(monkeypatch, exception):
    def fetch_batch(texts, timeout):
        raise exception

    monkeypatch.setattr(embeddings_module, "_fetch_batch", fetch_batch)


def _logged_traceback(record) -> str:
    """The exception text an operator would read off `record`.

    Formatted from exc_info rather than compared against str(exc): `exc_info=True`
    outside an except block attaches nothing, and that would still leave a
    non-null attribute behind.
    """
    assert record.exc_info is not None, "the handler logged no exception info"
    return "".join(traceback.format_exception(record.exc_info[1]))


# Level is the point of the table. WARNING for the transient-by-assumption
# classes and ERROR for the misconfiguration is a deliberate split: a
# misconfiguration never self-heals and maps to a 500, so at WARNING it would
# read exactly like routine backend flakiness and sit unnoticed in that noise.
# Driven through /api/v1/ingest because ingest maps all three; search degrades
# on the transient class instead (tests/test_search_degraded.py).
EMBEDDING_FAILURE_CASES = [
    pytest.param(
        EmbeddingBackendError, logging.WARNING, 503,
        "Embedding backend unavailable", True, id="transient",
    ),
    pytest.param(
        EmbeddingBackendBadResponse, logging.WARNING, 503,
        BAD_RESPONSE_DETAIL, True, id="bad-response",
    ),
    pytest.param(
        EmbeddingBackendMisconfigured, logging.ERROR, 500,
        "Embedding backend is misconfigured", False, id="misconfigured",
    ),
]


@pytest.mark.parametrize(
    "exception_class, level, status_code, detail, retry_after",
    EMBEDDING_FAILURE_CASES,
)
def test_an_embedding_backend_failure_is_logged_and_the_response_is_unchanged(
    http, db_session, monkeypatch, handler_records,
    exception_class, level, status_code, detail, retry_after,
):
    """The operator's only view of a backend fault, and the client's unchanged one.

    The response assertions sit here on purpose: a log line is the kind of
    change made without re-reading what the handler returns, and adding one
    must move nothing a caller sees. The format string itself is not asserted:
    rewording it is fine, dropping the class name or the path is not.
    """
    key = make_api_key(db_session, "acme")
    _fail_fetch_with(monkeypatch, exception_class(INTERNAL))

    response = http.post("/api/v1/ingest", json=INGEST_PAYLOAD, headers={"X-API-Key": key})

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    if retry_after:
        assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)
    else:
        assert "Retry-After" not in response.headers
    # The exception message is for logs; the response shows the fixed detail.
    assert INTERNAL not in response.text
    assert "internal-ollama" not in response.text

    records = handler_records()
    # Exactly one: logging twice doubles every alert this line feeds.
    assert len(records) == 1, records
    record = records[0]
    assert record.levelno == level
    # Without the class name a reader cannot tell a transient 503 from a 500
    # that will not recover without a deployment change.
    assert exception_class.__name__ in record.getMessage()
    assert "/api/v1/ingest" in record.getMessage()
    assert detail not in record.getMessage()
    # `is`, not isinstance: the subclasses carry different status codes, and a
    # traceback naming the parent would misdirect the reader.
    assert record.exc_info[0] is exception_class
    assert INTERNAL in _logged_traceback(record)


def test_the_logged_path_is_the_request_that_failed(
    http, db_session, monkeypatch, handler_records
):
    # The path is read off the request, so a second route is what rules out a
    # hardcoded string satisfying the test above. Search maps BadResponse (it
    # degrades only on the exact transient class).
    key = make_api_key(db_session, "acme")
    _fail_fetch_with(monkeypatch, EmbeddingBackendBadResponse("garbled"))

    response = http.post(
        "/api/v1/search", json={"q": "pelicanium"}, headers={"X-API-Key": key}
    )

    assert response.status_code == 503
    messages = [record.getMessage() for record in handler_records()]
    assert any("/api/v1/search" in m for m in messages), messages
    assert all("/api/v1/ingest" not in m for m in messages), messages


def test_a_degenerate_embedding_is_logged_with_the_component_that_was_wrong(
    http, db_session, monkeypatch, handler_records
):
    """The end-to-end case: a real NaN, through the real chain, into the log.

    The other logging tests stage the exception; this one lets _validated()
    compose the message, so the log carries what app.core.embeddings actually
    wrote while the client sees only the fixed detail.
    """
    key = make_api_key(db_session, "acme")
    vector = [0.01] * settings.embedding_dimension
    vector[0] = float("nan")
    monkeypatch.setattr(
        embeddings_module, "_fetch_batch", lambda texts, timeout: [vector for _ in texts]
    )

    response = http.post(
        "/api/v1/search", json={"q": "pelicanium"}, headers={"X-API-Key": key}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": BAD_RESPONSE_DETAIL}
    assert response.headers["Retry-After"] == str(RETRY_AFTER_SECONDS)

    records = handler_records()
    assert len(records) == 1, records
    logged = _logged_traceback(records[0])
    assert "nan" in logged
    assert "expected finite float32 values" in logged
    assert "nan" not in response.text


def test_health_is_unaffected(client):
    # /health contacts nothing, so a dependency outage cannot turn a liveness
    # probe into a restart.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
