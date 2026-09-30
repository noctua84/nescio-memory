"""Unit tests for get_embedding's failure translation.

No database and no container: these call get_embedding directly with a stubbed
transport, so a failure here points at the embedding module rather than at the
HTTP layer.
"""
import httpx
import pytest

from app.core import embeddings as embeddings_module
from app.core.embeddings import get_embedding
from app.core.errors import (
    EmbeddingBackendBadResponse,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)


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
    with pytest.raises(EmbeddingBackendError) as excinfo:
        get_embedding("anything")
    # Exact type, not isinstance: pytest.raises matches subclasses, so asserting
    # the base class alone would not catch a regression that raised
    # EmbeddingBackendBadResponse for a connection failure.
    assert type(excinfo.value) is EmbeddingBackendError


def test_timeout_raises_embedding_backend_error(monkeypatch):
    def time_out():
        raise httpx.ReadTimeout("too slow")

    _install(monkeypatch, time_out)
    with pytest.raises(EmbeddingBackendError) as excinfo:
        get_embedding("anything")
    # Exact type, not isinstance: pytest.raises matches subclasses, so asserting
    # the base class alone would not catch a regression that raised
    # EmbeddingBackendBadResponse for a timeout.
    assert type(excinfo.value) is EmbeddingBackendError


def test_non_2xx_raises_embedding_backend_error(monkeypatch):
    _install(monkeypatch, lambda: _response({"error": "boom"}, status_code=500))
    with pytest.raises(EmbeddingBackendError) as excinfo:
        get_embedding("anything")
    # Exact type, not isinstance: pytest.raises matches subclasses, so asserting
    # the base class alone would not catch a regression that raised
    # EmbeddingBackendBadResponse for a non-2xx response.
    assert type(excinfo.value) is EmbeddingBackendError


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
    vector = [0.01] * 384
    _install(monkeypatch, lambda: _response({"embedding": vector}))
    assert get_embedding("anything") == vector


def test_a_malformed_ollama_url_is_reported_as_misconfiguration(monkeypatch):
    def invalid_url():
        raise httpx.InvalidURL("no scheme supplied")

    _install(monkeypatch, invalid_url)
    with pytest.raises(EmbeddingBackendMisconfigured):
        get_embedding("anything")


def test_a_local_backend_runtime_failure_is_treated_as_transient(monkeypatch):
    class _OutOfMemory:
        def encode(self, text):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(embeddings_module.settings, "embedding_backend", "local")
    monkeypatch.setattr(embeddings_module, "_get_local_model", lambda: _OutOfMemory())

    with pytest.raises(EmbeddingBackendError) as excinfo:
        get_embedding("anything")
    # Transient, not misconfiguration: a later attempt may succeed, so this must
    # map to 503 rather than 500.
    assert type(excinfo.value) is EmbeddingBackendError


def test_a_missing_local_extra_stays_a_misconfiguration(monkeypatch):
    def missing_extra():
        raise EmbeddingBackendMisconfigured("local-embeddings extra not installed")

    monkeypatch.setattr(embeddings_module.settings, "embedding_backend", "local")
    monkeypatch.setattr(embeddings_module, "_get_local_model", missing_extra)

    # Guards the re-raise clause: without it, the broad `except Exception` would
    # downgrade this permanent fault to a transient 503.
    with pytest.raises(EmbeddingBackendMisconfigured):
        get_embedding("anything")


@pytest.mark.parametrize(
    "bad_value, why",
    [
        ([], "empty"),
        ([0.1] * 768, "wrong dimension"),
        ("notavector", "a string"),
        (None, "null"),
        ([None] * 384, "a list of nulls"),
        ({"a": 1}, "an object"),
        ([True] * 384, "booleans"),
    ],
)
def test_an_unusable_embedding_value_is_rejected(monkeypatch, bad_value, why):
    # The key is present in every case, so the key-existence check passes and
    # only the shape check can catch these.
    _install(monkeypatch, lambda: _response({"embedding": bad_value}))
    with pytest.raises(EmbeddingBackendBadResponse):
        get_embedding(f"a response carrying {why}")


def test_a_correctly_shaped_embedding_still_passes(monkeypatch):
    vector = [0.01] * 384
    _install(monkeypatch, lambda: _response({"embedding": vector}))
    assert get_embedding("anything") == vector
