"""Startup validation of settings.

These construct Settings directly rather than importing the module-level
singleton, so a bad value fails here rather than at import time.
"""
import pytest

from app.config import Settings


def _settings(**overrides):
    base = {
        "database_url": "postgresql+psycopg://u:p@localhost:5432/db",
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


@pytest.mark.parametrize("in_range", [1, 60000])
def test_the_statement_timeout_accepts_both_ends_of_its_range(in_range):
    # 1ms is useless but coherent; 60000ms is the documented ceiling, and a
    # ceiling that rejects its own boundary value is a different ceiling.
    assert _settings(statement_timeout_ms=in_range).statement_timeout_ms == in_range


@pytest.mark.parametrize("too_large", [60001, 5_000_000])
def test_a_statement_timeout_above_the_ceiling_fails_at_startup(too_large):
    # 5_000_000 is the typo this ceiling exists to catch: a misplaced digit
    # reinstates the effectively unbounded search of issue #17.
    with pytest.raises(ValueError, match="STATEMENT_TIMEOUT_MS"):
        _settings(statement_timeout_ms=too_large)


@pytest.mark.parametrize("rejected", [0, -1, 60001, 5_000_000])
def test_the_statement_timeout_error_states_the_permitted_range(rejected):
    # An operator who sees this message must learn the bound, not just that
    # their value was refused.
    with pytest.raises(ValueError, match=r"1\.\.60000"):
        _settings(statement_timeout_ms=rejected)


def test_db_connect_timeout_s_defaults_to_10():
    assert _settings().db_connect_timeout_s == 10


def test_a_db_connect_timeout_of_2_is_accepted():
    # 2 is the floor, not a rejected value: it is the smallest timeout libpq
    # actually applies as written.
    assert _settings(db_connect_timeout_s=2).db_connect_timeout_s == 2


def test_a_larger_db_connect_timeout_is_accepted():
    assert _settings(db_connect_timeout_s=30).db_connect_timeout_s == 30


@pytest.mark.parametrize("below_floor", [1, 0, -1])
def test_a_db_connect_timeout_below_2_fails_at_startup(below_floor):
    # libpq silently raises anything below 2 to 2, so accepting 1 would claim
    # a bound that is never applied; 0 disables the timeout entirely.
    with pytest.raises(ValueError, match="DB_CONNECT_TIMEOUT_S"):
        _settings(db_connect_timeout_s=below_floor)


def test_db_connect_timeout_reads_its_environment_variable(monkeypatch):
    # The name an operator sets in the environment must be the name the
    # setting binds to, independent of any .env file on disk.
    monkeypatch.setenv("DB_CONNECT_TIMEOUT_S", "17")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db"
    )
    assert Settings(_env_file=None).db_connect_timeout_s == 17


def test_a_bad_db_connect_timeout_in_the_environment_fails_at_startup(monkeypatch):
    monkeypatch.setenv("DB_CONNECT_TIMEOUT_S", "1")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db"
    )
    with pytest.raises(ValueError, match="DB_CONNECT_TIMEOUT_S"):
        Settings(_env_file=None)


def test_max_context_documents_defaults_to_25():
    assert _settings().max_context_documents == 25


def test_a_single_context_document_is_accepted():
    assert _settings(max_context_documents=1).max_context_documents == 1


@pytest.mark.parametrize("non_positive", [0, -1])
def test_a_non_positive_max_context_documents_fails_at_startup(non_positive):
    # 0 does not narrow context expansion, it disables it while /search still
    # accepts requests asking for context -- indistinguishable from a bug.
    with pytest.raises(ValueError, match="MAX_CONTEXT_DOCUMENTS"):
        _settings(max_context_documents=non_positive)


def test_max_context_documents_reads_its_environment_variable(monkeypatch):
    monkeypatch.setenv("MAX_CONTEXT_DOCUMENTS", "7")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db"
    )
    assert Settings(_env_file=None).max_context_documents == 7
