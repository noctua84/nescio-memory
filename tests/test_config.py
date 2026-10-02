"""Startup validation of settings.

These construct Settings directly rather than importing the module-level
singleton, so a bad value fails here rather than at import time.
"""
import pytest

from app.config import Settings


def _settings(**overrides):
    base = {
        "database_url": "postgresql+psycopg2://u:p@localhost:5432/db",
        "ollama_url": "http://localhost:11434/api/embeddings",
    }
    base.update(overrides)
    return Settings(**base)


@pytest.mark.parametrize(
    "bad_url",
    ["", "not a url at all", "localhost:11434/api/embeddings", "/api/embeddings",
     "ftp://host/x"],
)
def test_a_malformed_ollama_url_fails_at_startup(bad_url):
    # Each of these otherwise surfaces per request as a 503 promising a retry
    # that can never succeed.
    with pytest.raises(ValueError, match="OLLAMA_URL"):
        _settings(ollama_url=bad_url)


def test_a_valid_ollama_url_is_accepted():
    assert _settings(ollama_url="https://ollama.internal:11434/api/embeddings")


def test_the_local_backend_does_not_require_a_valid_ollama_url():
    # The local backend never reads ollama_url, so it must not be forced to
    # supply one.
    assert _settings(embedding_backend="local", ollama_url="")


def test_statement_timeout_ms_defaults_to_5000():
    assert _settings().statement_timeout_ms == 5000


@pytest.mark.parametrize("non_positive", [0, -1, -5000])
def test_a_non_positive_statement_timeout_fails_at_startup(non_positive):
    # 0 would disable the timeout in Postgres and a negative value is rejected
    # by set_config; both would silently remove the cap issue #17 added.
    with pytest.raises(ValueError, match="STATEMENT_TIMEOUT_MS"):
        _settings(statement_timeout_ms=non_positive)


def test_a_positive_statement_timeout_is_accepted():
    assert _settings(statement_timeout_ms=1234).statement_timeout_ms == 1234
