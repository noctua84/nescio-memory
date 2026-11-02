"""The read-back guard in PassageSearchRepository.search checks EACH setting.

The search applies hnsw.iterative_scan (#12), hnsw.max_scan_tuples (#12) and
statement_timeout (#15/#17) with set_config(..., true) -- SET LOCAL -- and
reads all three back, raising RuntimeError if any did not take effect. Under
AUTOCOMMIT there is no transaction block, so SET LOCAL covers only its own
statement and the search would otherwise run with defaults while answering 200.

Already pinned elsewhere, and not repeated here:
- the guard fires under AUTOCOMMIT at all, on the FTS-only and hybrid paths:
  tests/test_search_hybrid_guards.py::test_read_back_guard_fires_without_an_open_transaction
  (and the FTS-only case in tests/test_search_contract.py);
- it does NOT fire as the first statement of an ordinary Session, and the
  settings do not outlive the transaction:
  tests/test_search_hybrid_guards.py::test_guards_do_not_outlive_the_search_transaction;
- Postgres's normalised timeout spellings ("5s", "1min") are accepted:
  tests/test_search_hybrid_guards.py::test_guard_accepts_postgres_normalised_timeout_spellings.

The distinct assertion here: those AUTOCOMMIT tests lose all three settings at
once, so they would still pass if the guard checked only the first one. Here
two of the three are put in force at SESSION level on the AUTOCOMMIT
connection (plain SET persists there), leaving exactly one that SET LOCAL
cannot deliver -- and the guard must name that one. A control with all three
pre-set shows the guard compares values rather than inspecting the
transaction state.

These tests deliberately avoid the db_session fixture: it wraps everything in
an outer transaction plus a savepoint, where SET LOCAL always persists. A
session-level SET would outlive the test on a pooled connection, so every
connection used here is invalidated rather than returned to the pool. Nothing
here writes rows.
"""
import re
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

import app.config as config_module
from app.repositories.passage_search import PassageSearchRepository

# Distinctive, so a pre-set value cannot be confused with a default; Postgres
# renders it back unchanged as "4321ms".
_TIMEOUT_MS = 4321

# What the guard expects of each setting, as a session-level SET statement.
_SESSION_SETS = {
    "hnsw.iterative_scan": "SET hnsw.iterative_scan = 'strict_order'",
    "hnsw.max_scan_tuples": "SET hnsw.max_scan_tuples = 100000",
    "statement_timeout": f"SET statement_timeout = {_TIMEOUT_MS}",
}


@contextmanager
def _autocommit_session(engine, preset: list[str]) -> Iterator[Session]:
    """A Session on an AUTOCOMMIT connection with `preset` settings already in
    force at session level, discarded afterwards so they cannot leak."""
    connection = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    session = Session(bind=connection)
    try:
        # Load the vector library first, so the hnsw.* SETs below land on the
        # real parameters rather than on placeholders.
        connection.execute(text("SELECT '[1]'::vector"))
        for name in preset:
            connection.execute(text(_SESSION_SETS[name]))
        yield session
    finally:
        session.close()
        # Session-level SETs are not undone by the pool's reset-on-return.
        connection.invalidate()
        connection.close()


@pytest.mark.parametrize("missing", list(_SESSION_SETS))
def test_the_guard_names_the_one_setting_that_did_not_take_effect(
    engine, monkeypatch, missing
):
    monkeypatch.setattr(config_module.settings, "statement_timeout_ms", _TIMEOUT_MS)
    preset = [name for name in _SESSION_SETS if name != missing]

    with _autocommit_session(engine, preset) as session:
        repository = PassageSearchRepository(session, "acme")
        with pytest.raises(RuntimeError) as exc_info:
            repository.search("pelicanium", None, 5)

    message = str(exc_info.value)
    assert re.match(rf"{re.escape(missing)} setting did not take effect", message)
    for other in preset:
        assert other not in message


def test_the_guard_passes_under_autocommit_when_every_setting_is_in_force(
    engine, monkeypatch
):
    # Control for the test above: with all three already in force the read-back
    # matches, so it is the missing value -- not AUTOCOMMIT itself, nor any
    # other side effect of the session-level SETs -- that the guard rejects.
    monkeypatch.setattr(config_module.settings, "statement_timeout_ms", _TIMEOUT_MS)

    with _autocommit_session(engine, list(_SESSION_SETS)) as session:
        rows = PassageSearchRepository(session, "acme").search("pelicanium", None, 5)

    assert list(rows) == []
