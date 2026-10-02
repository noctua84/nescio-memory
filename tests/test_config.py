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


def test_a_bare_postgresql_url_is_rewritten_to_psycopg2():
    # SQLAlchemy 2.1 defaults to psycopg 3, but this project only ships
    # psycopg2; the bare scheme is normalized at startup.
    s = _settings(database_url="postgresql://user:pass@localhost:5432/db")
    assert s.database_url == "postgresql+psycopg2://user:pass@localhost:5432/db"


def test_a_postgresql_plus_psycopg2_url_is_unchanged():
    # Explicit driver schemes are not modified.
    url = "postgresql+psycopg2://user:pass@localhost:5432/db"
    s = _settings(database_url=url)
    assert s.database_url == url


def test_a_postgresql_plus_psycopg_url_is_unchanged():
    # Explicit driver schemes are not modified.
    url = "postgresql+psycopg://user:pass@localhost:5432/db"
    s = _settings(database_url=url)
    assert s.database_url == url
