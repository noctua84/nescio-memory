"""The warning POST /api/v1/search logs when it degrades to full-text.

It names the exception class AND carries the exception message: without the
message, the log cannot tell a timeout from a refused connection from a 503.
That is safe only because no real query-path message echoes the query, so the
second test drives each real path -- Ollama status, transport errors, the local
model's runtime error -- with a distinctive query and asserts it never reaches
any app log line. Nothing here reaches the network.
"""
import logging

import httpx
import pytest

from app.api.v1 import search as search_module
from app.config import settings
from app.core import embeddings as embeddings_module
from app.core.errors import EmbeddingBackendError
from tests.factories import make_api_key, make_note, make_passage

# Distinctive enough that finding either word in a log line can only mean the
# query text leaked there. "pelicanium" also lets FTS find the seeded passage.
SECRET_QUERY = "pelicanium quokkasecret"
OLLAMA_URL = "http://embedder.invalid:11434/api/embed"


@pytest.fixture
def search_log(client, db_session, monkeypatch, caplog):
    """Seed one passage, then return a function that searches and yields the
    degraded body plus every app log message emitted during the request."""
    # alembic/env.py's fileConfig() disables every logger that already exists
    # when the engine fixture migrates (see tests/test_error_responses.py).
    monkeypatch.setattr(search_module.logger, "disabled", False)
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme")
    make_passage(
        db_session, note, ordinal=0, content="The pelicanium isotope decays."
    )

    def run(q=SECRET_QUERY):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=search_module.logger.name):
            response = client.post(
                "/api/v1/search",
                json={"q": q, "top_k": 10},
                headers={"X-API-Key": key},
            )
        assert response.status_code == 200, response.text
        route = [r for r in caplog.records if r.name == search_module.logger.name]
        app = [r.getMessage() for r in caplog.records if r.name.startswith("app")]
        return response.json(), route, app

    return run


def test_the_degraded_warning_names_the_class_and_the_exception_message(
    search_log, monkeypatch
):
    def unavailable(query):
        raise EmbeddingBackendError("Ollama request failed: HTTP 503")

    monkeypatch.setattr(search_module, "get_query_embedding", unavailable)

    body, route_records, _ = search_log()

    assert body["degraded"] is True
    assert len(route_records) == 1
    record = route_records[0]
    assert record.levelno == logging.WARNING
    assert record.getMessage() == (
        "query embedding unavailable (EmbeddingBackendError: Ollama request "
        "failed: HTTP 503); serving full-text results only"
    )


class _StubClient:
    """Stands in for httpx.Client; every post() answers with `behaviour()`."""

    def __init__(self, behaviour):
        self._behaviour = behaviour
        self.posts = 0

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        self.posts += 1
        return self._behaviour()


def _ollama_503():
    return httpx.Response(
        status_code=503,
        json={"error": "server busy"},
        request=httpx.Request("POST", OLLAMA_URL),
    )


def _ollama_refused():
    raise httpx.ConnectError(
        "[Errno 111] Connection refused", request=httpx.Request("POST", OLLAMA_URL)
    )


def _ollama_timeout():
    raise httpx.ReadTimeout("timed out", request=httpx.Request("POST", OLLAMA_URL))


class _FailingLocalModel:
    """A local model whose runtime error, like many real ones, echoes its input."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        raise RuntimeError(f"cannot encode {texts!r}: out of memory")


@pytest.mark.parametrize(
    "backend, failure, expected_fragment",
    [
        ("ollama", _ollama_503, "Ollama request failed: HTTP 503"),
        ("ollama", _ollama_refused, "Connection refused"),
        ("ollama", _ollama_timeout, "timed out"),
        ("local", None, "RuntimeError"),
    ],
    ids=["ollama-503", "ollama-connect-error", "ollama-read-timeout", "local-oom"],
)
def test_the_degraded_warning_never_carries_the_query_on_a_real_path(
    search_log, monkeypatch, backend, failure, expected_fragment
):
    # The real get_query_embedding chain, not the `client` fixture's fake.
    monkeypatch.setattr(
        search_module, "get_query_embedding", embeddings_module.get_query_embedding
    )
    monkeypatch.setattr(settings, "embedding_backend", backend)
    monkeypatch.setattr(settings, "ollama_url", OLLAMA_URL)
    if backend == "ollama":
        reached = _StubClient(failure)
        monkeypatch.setattr(embeddings_module.httpx, "Client", reached)
    else:
        reached = _FailingLocalModel()
        monkeypatch.setattr(embeddings_module, "_get_local_model", lambda: reached)

    body, route_records, app_messages = search_log()

    assert body["degraded"] is True
    assert (reached.posts if backend == "ollama" else reached.calls) == 1, (
        "the stubbed backend was never reached"
    )
    assert len(route_records) == 1
    message = route_records[0].getMessage()
    # The message really is logged, so its absence of the query is meaningful.
    assert "EmbeddingBackendError" in message
    assert expected_fragment in message
    assert all("quokkasecret" not in m for m in app_messages)
    assert all("pelicanium" not in m for m in app_messages)
