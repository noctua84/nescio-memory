"""Startup validation of settings.

These construct Settings directly rather than importing the module-level
singleton, so a bad value fails here rather than at import time.
"""
import math
import re

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
    ["", "not a url at all", "localhost:11434/api/embed", "/api/embed",
     "ftp://host/x"],
)
def test_a_malformed_ollama_url_fails_at_startup(bad_url):
    # Each of these otherwise surfaces per request as a 503 promising a retry
    # that can never succeed.
    with pytest.raises(ValueError, match="OLLAMA_URL"):
        _settings(ollama_url=bad_url)


def test_a_valid_ollama_url_is_accepted():
    assert _settings(ollama_url="https://ollama.internal:11434/api/embed")


def test_the_ollama_url_defaults_to_the_batch_endpoint():
    # Read off the field rather than a constructed Settings, so an OLLAMA_URL
    # in the environment cannot mask the default.
    assert Settings.model_fields["ollama_url"].default == (
        "http://localhost:11434/api/embed"
    )


@pytest.mark.parametrize(
    "legacy_url",
    [
        "http://localhost:11434/api/embeddings",
        "http://localhost:11434/api/embeddings/",
        "https://ollama.internal:11434/api/embeddings",
    ],
)
def test_the_legacy_embeddings_endpoint_fails_at_startup(legacy_url):
    # Ollama still answers the legacy endpoint with a different shape, so a
    # stale .env would otherwise fail every request as a retryable 503.
    with pytest.raises(ValueError) as excinfo:
        _settings(ollama_url=legacy_url)
    message = str(excinfo.value)
    assert "OLLAMA_URL" in message
    # The remedy must be named. The lookahead stops "/api/embeddings" in the
    # same message from satisfying this by itself.
    assert re.search(r"/api/embed(?!dings)", message)


def test_the_legacy_endpoint_is_ignored_by_the_local_backend():
    # The local backend never reads OLLAMA_URL, so a stale value is harmless.
    assert _settings(
        embedding_backend="local", ollama_url="http://localhost:11434/api/embeddings"
    )


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


@pytest.mark.parametrize(
    "stale_env_var, stale_field",
    [
        ("CHUNK_SIZE", "chunk_size"),
        ("CHUNK_OVERLAP", "chunk_overlap"),
        ("MAX_CONTEXT_CHARS", "max_context_chars"),
        ("MAX_CONTEXT_DOCUMENTS", "max_context_documents"),
    ],
)
def test_an_env_var_for_a_removed_setting_is_ignored(
    monkeypatch, stale_env_var, stale_field
):
    # These four settings went with the removed learnings API. A deployment whose
    # .env still carries them must keep booting, so Settings relies on
    # extra="ignore": the stale variable is neither rejected as an unknown
    # field nor bound to an attribute anything could still read. A value that
    # the removed validators would have refused (0, an overlap that never
    # advances the window) makes a surviving validator impossible to miss.
    monkeypatch.setenv(stale_env_var, "0")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db")
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434/api/embed")

    loaded = Settings(_env_file=None)

    assert not hasattr(loaded, stale_field)
    assert stale_field not in Settings.model_fields


def test_stale_settings_are_ignored_when_passed_as_init_kwargs():
    # The same contract for values that reach Settings as keyword arguments
    # (the route _settings() above takes), which pydantic validates separately
    # from the environment source.
    loaded = _settings(chunk_size=1000, chunk_overlap=200, max_context_chars=0)

    for stale_field in ("chunk_size", "chunk_overlap", "max_context_chars"):
        assert not hasattr(loaded, stale_field)


# --- search and query-embedding settings ---
#
# Each validator is pinned from both sides of each bound: a just-invalid value
# that must fail naming its env var, and a just-valid one that must pass. A
# bound tested only from one side would survive an off-by-one.

NEW_SETTING_DEFAULTS = {
    "search_candidates": 50,
    "rrf_k": 60,
    "fts_weight": 1.0,
    "vector_weight": 1.0,
    "query_instruction": (
        "Given a web search query, retrieve relevant passages that answer the query"
    ),
    "query_embed_timeout_s": 5.0,
    "embed_batch_size": 16,
}


@pytest.mark.parametrize("field, default", NEW_SETTING_DEFAULTS.items())
def test_a_new_setting_has_its_documented_default(field, default):
    # Off the field, so a value in the environment cannot mask the default.
    assert Settings.model_fields[field].default == default


@pytest.mark.parametrize(
    "field, env_var, rejected",
    [
        ("search_candidates", "SEARCH_CANDIDATES", 0),
        ("search_candidates", "SEARCH_CANDIDATES", 1001),
        ("rrf_k", "RRF_K", 0),
        ("rrf_k", "RRF_K", -1),
        ("fts_weight", "FTS_WEIGHT", -0.001),
        ("fts_weight", "FTS_WEIGHT", math.nan),
        ("fts_weight", "FTS_WEIGHT", math.inf),
        ("fts_weight", "FTS_WEIGHT", -math.inf),
        ("vector_weight", "VECTOR_WEIGHT", -0.001),
        ("vector_weight", "VECTOR_WEIGHT", math.nan),
        ("vector_weight", "VECTOR_WEIGHT", math.inf),
        ("vector_weight", "VECTOR_WEIGHT", -math.inf),
        ("query_embed_timeout_s", "QUERY_EMBED_TIMEOUT_S", 0),
        ("query_embed_timeout_s", "QUERY_EMBED_TIMEOUT_S", -1),
        ("query_embed_timeout_s", "QUERY_EMBED_TIMEOUT_S", 30.001),
        ("query_embed_timeout_s", "QUERY_EMBED_TIMEOUT_S", math.nan),
        ("query_embed_timeout_s", "QUERY_EMBED_TIMEOUT_S", math.inf),
        ("embed_batch_size", "EMBED_BATCH_SIZE", 0),
        ("embed_batch_size", "EMBED_BATCH_SIZE", 257),
    ],
)
def test_an_out_of_range_setting_fails_at_startup_naming_its_env_var(
    field, env_var, rejected
):
    # NaN compares false against every bound, so a range check written as
    # `x < 0` alone would let it through; it is listed wherever finiteness is
    # part of the contract.
    with pytest.raises(ValueError, match=env_var):
        _settings(**{field: rejected})


@pytest.mark.parametrize(
    "field, accepted",
    [
        ("search_candidates", 1),
        ("search_candidates", 1000),
        ("rrf_k", 1),
        ("fts_weight", 0.0),
        ("vector_weight", 0.0),
        ("query_embed_timeout_s", 0.001),
        ("query_embed_timeout_s", 30),
        ("embed_batch_size", 1),
        ("embed_batch_size", 256),
        ("query_instruction", ""),
    ],
)
def test_an_in_range_setting_is_accepted(field, accepted):
    # One weight at 0 is how a single-arm baseline is measured, and an empty
    # instruction is how a model without an instruction format is served.
    assert getattr(_settings(**{field: accepted}), field) == accepted


def test_both_weights_at_zero_fail_at_startup():
    # With both arms off every search would return nothing.
    with pytest.raises(ValueError) as excinfo:
        _settings(fts_weight=0.0, vector_weight=0.0)
    message = str(excinfo.value)
    assert "FTS_WEIGHT" in message
    assert "VECTOR_WEIGHT" in message


@pytest.mark.parametrize(
    "env_var, raw, field, expected",
    [
        ("SEARCH_CANDIDATES", "77", "search_candidates", 77),
        ("RRF_K", "12", "rrf_k", 12),
        ("FTS_WEIGHT", "0.25", "fts_weight", 0.25),
        ("VECTOR_WEIGHT", "2.5", "vector_weight", 2.5),
        ("QUERY_INSTRUCTION", "Find notes", "query_instruction", "Find notes"),
        ("QUERY_EMBED_TIMEOUT_S", "7.5", "query_embed_timeout_s", 7.5),
        ("EMBED_BATCH_SIZE", "32", "embed_batch_size", 32),
    ],
)
def test_a_new_setting_reads_its_environment_variable(
    monkeypatch, env_var, raw, field, expected
):
    # The name an operator sets must be the name the setting binds to, and
    # the name every error message above refers them to.
    monkeypatch.setenv(env_var, raw)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db")
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434/api/embed")
    assert getattr(Settings(_env_file=None), field) == expected
