"""Unit tests for get_embedding's failure translation.

No database and no container: these call get_embedding directly with a stubbed
transport, so a failure here points at the embedding module rather than at the
HTTP layer.
"""
import httpx
import pytest

from app.core import embeddings as embeddings_module
from app.core.embeddings import get_embedding
from app.core.errors import EmbeddingBackendBadResponse, EmbeddingBackendError


class _StubClient:
    """Stands in for httpx.Client. `behaviour` decides what post() does."""

    def __init__(self, behaviour):
        self._behaviour = behaviour

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return self._behaviour()


def _install(monkeypatch, behaviour):
    monkeypatch.setattr(embeddings_module.httpx, "Client", _StubClient(behaviour))


def _response(payload, status_code=200):
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("POST", "http://ollama.invalid/api/embeddings"),
    )


def test_connection_failure_raises_embedding_backend_error(monkeypatch):
    def refuse():
        raise httpx.ConnectError("connection refused")

    _install(monkeypatch, refuse)
    with pytest.raises(EmbeddingBackendError):
        get_embedding("anything")


def test_timeout_raises_embedding_backend_error(monkeypatch):
    def time_out():
        raise httpx.ReadTimeout("too slow")

    _install(monkeypatch, time_out)
    with pytest.raises(EmbeddingBackendError):
        get_embedding("anything")


def test_non_2xx_raises_embedding_backend_error(monkeypatch):
    _install(monkeypatch, lambda: _response({"error": "boom"}, status_code=500))
    with pytest.raises(EmbeddingBackendError):
        get_embedding("anything")


def test_missing_embedding_key_raises_bad_response(monkeypatch):
    _install(monkeypatch, lambda: _response({"not_what_we_expected": []}))
    with pytest.raises(EmbeddingBackendBadResponse):
        get_embedding("anything")


def test_non_json_body_raises_bad_response(monkeypatch):
    def html_page():
        return httpx.Response(
            status_code=200,
            text="<html>proxy error</html>",
            request=httpx.Request("POST", "http://ollama.invalid/api/embeddings"),
        )

    _install(monkeypatch, html_page)
    with pytest.raises(EmbeddingBackendBadResponse):
        get_embedding("anything")


def test_bad_response_is_a_subclass_so_one_handler_covers_both(monkeypatch):
    # A caller that does not care which kind of failure occurred should be able
    # to catch the base class alone.
    assert issubclass(EmbeddingBackendBadResponse, EmbeddingBackendError)


def test_a_successful_call_still_returns_the_vector(monkeypatch):
    _install(monkeypatch, lambda: _response({"embedding": [0.1, 0.2, 0.3]}))
    assert get_embedding("anything") == [0.1, 0.2, 0.3]
