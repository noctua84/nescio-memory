"""OLLAMA_URL must end in /api/embed.

The legacy-endpoint and trailing-slash rejections (with their own, more specific
messages) are covered in tests/test_config.py; this file covers the remaining
case: a URL that is well-formed but points somewhere other than the batch
endpoint, notably a bare host.
"""
import pytest

from app.config import Settings


def _settings(**overrides):
    base = {
        "database_url": "postgresql+psycopg://u:p@localhost:5432/db",
        "ollama_url": "http://localhost:11434/api/embed",
    }
    base.update(overrides)
    return Settings(**base)


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://localhost:11434",
        "https://ollama.internal",
        "http://localhost:11434/api/generate",
        "http://localhost:11434/api/embedding",
        "http://localhost:11434/v1/api/embed/extra",
    ],
)
def test_a_url_that_is_not_the_batch_endpoint_fails_at_startup(bad_url):
    # A bare host otherwise posts to the server root and fails per request.
    with pytest.raises(ValueError) as excinfo:
        _settings(ollama_url=bad_url)
    message = str(excinfo.value)
    assert "OLLAMA_URL" in message
    assert "/api/embed" in message


@pytest.mark.parametrize(
    "good_url",
    [
        "http://localhost:11434/api/embed",
        "http://proxy.internal/ollama/api/embed",
        "https://gateway.internal:8443/team/a/ollama/api/embed",
    ],
)
def test_the_batch_endpoint_with_or_without_a_proxy_prefix_is_accepted(good_url):
    assert _settings(ollama_url=good_url).ollama_url == good_url


def test_the_local_backend_ignores_a_bare_host():
    # The local backend never reads OLLAMA_URL, so a stale value is harmless.
    assert _settings(embedding_backend="local", ollama_url="http://localhost:11434")
