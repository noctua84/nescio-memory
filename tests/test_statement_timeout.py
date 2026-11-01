"""The search-scoped statement_timeout, end to end: a real cancellation.

Where the neighbouring concerns live, so they are not repeated here:
- startup validation of STATEMENT_TIMEOUT_MS: tests/test_config.py;
- that PassageSearchRepository.search applies settings.statement_timeout_ms
  via SET LOCAL (and Postgres's normalised spellings of it):
  tests/test_search_contract.py and tests/test_search_hybrid_guards.py;
- that the setting does not outlive the search transaction on a pooled
  connection: tests/test_search_hybrid_guards.py;
- the 57014 -> 503 mapping with a FAKED QueryCanceled (and its log line):
  tests/test_error_responses_database.py.

What only this file proves (#15/#17): with a tight timeout, a passages search
that runs long is genuinely canceled BY POSTGRES, the driver surfaces that as
OperationalError wrapping psycopg's QueryCanceled (SQLSTATE 57014) -- the exact
shape app.core.errors keys off -- and the route answers 503 with no
Retry-After, because a timeout is deterministic for the same input.

How the search is made slow: a second connection holds ACCESS EXCLUSIVE on
`passages`, so the search's own SELECT waits on the lock. statement_timeout
covers lock waits, so the unmodified production statement is what gets
canceled -- no SQL is rewritten and no corpus has to be large enough to be
slow on every machine. The guard statements (set_config / current_setting)
touch no table and run before the wait, so the lock delays only the search.

Every lock wait is bounded twice so a regression fails instead of hanging: the
search side carries a lock_timeout well above the statement_timeout under
test, so if the statement_timeout were ever not applied the wait would end in
SQLSTATE 55P03 (lock not available), which these assertions reject.
"""
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg.errors
import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

import app.config as config_module
from app.core.errors import QUERY_TIMEOUT_DETAIL
from app.repositories.passage_search import PassageSearchRepository
from tests.factories import make_api_key
from tests.fakes import unit_vector

# Tight enough that the test is quick, far enough below _HANG_BOUND that the
# two cannot be confused.
_TIGHT_TIMEOUT_MS = 200
_HANG_BOUND = "5s"


@contextmanager
def _passages_locked(engine) -> Iterator[None]:
    """Hold ACCESS EXCLUSIVE on `passages` from a separate connection until exit.

    Its own connection, rolled back in `finally`: that rollback is what
    releases the lock, and it keeps this transaction away from the shared
    `connection` fixture's. lock_timeout bounds acquiring the lock itself.
    """
    holder = engine.connect()
    transaction = holder.begin()
    try:
        holder.execute(text(f"SET LOCAL lock_timeout = '{_HANG_BOUND}'"))
        holder.execute(text("LOCK TABLE passages IN ACCESS EXCLUSIVE MODE"))
        yield
    finally:
        transaction.rollback()
        holder.close()


def _bound_the_wait(db_session) -> None:
    # Runs inside the test's outer transaction, so the rollback undoes it.
    db_session.execute(text(f"SET LOCAL lock_timeout = '{_HANG_BOUND}'"))


@pytest.mark.parametrize("query_vector", [None, unit_vector(0)],
                         ids=["fts-only", "hybrid"])
def test_a_slow_passages_search_is_canceled_with_sqlstate_57014(
    engine, db_session, monkeypatch, query_vector
):
    # Both paths: the guard is applied unconditionally, so an FTS-only search
    # (degraded mode, or VECTOR_WEIGHT=0) is capped exactly like a hybrid one.
    monkeypatch.setattr(config_module.settings, "statement_timeout_ms",
                        _TIGHT_TIMEOUT_MS)
    _bound_the_wait(db_session)
    repo = PassageSearchRepository(db_session, "acme")

    with _passages_locked(engine), pytest.raises(OperationalError) as exc_info:
        repo.search("pelicanium", query_vector, 5)

    assert isinstance(exc_info.value.orig, psycopg.errors.QueryCanceled)
    assert exc_info.value.orig.sqlstate == "57014"
    # The canceled statement is the search itself, not one of the guard
    # statements that set and read back the timeout.
    assert "passages" in exc_info.value.statement
    assert "set_config" not in exc_info.value.statement
    assert "current_setting" not in exc_info.value.statement


def test_a_search_canceled_by_its_timeout_returns_503_with_no_retry_after(
    client, engine, db_session, monkeypatch
):
    # The same real cancellation, through POST /api/v1/search. 503 with no
    # Retry-After: the timeout is deterministic for the same input, so a retry
    # hint would be untrue -- unlike the generic OperationalError branch,
    # which does send one.
    key = make_api_key(db_session, "acme")
    monkeypatch.setattr(config_module.settings, "statement_timeout_ms",
                        _TIGHT_TIMEOUT_MS)
    _bound_the_wait(db_session)

    with _passages_locked(engine):
        response = client.post(
            "/api/v1/search",
            json={"q": "pelicanium", "top_k": 5},
            headers={"X-API-Key": key},
        )

    assert response.status_code == 503, response.text
    assert response.json() == {"detail": QUERY_TIMEOUT_DETAIL}
    assert "Retry-After" not in response.headers


def test_postgres_actually_cancels_a_statement_over_a_tight_timeout(engine):
    # The driver/server behaviour in isolation, with no repository involved:
    # a tight statement_timeout raises OperationalError whose .orig is a
    # psycopg QueryCanceled carrying SQLSTATE 57014, which is exactly what
    # app.core.errors keys off via isinstance(). If this fails, so will the
    # two tests above, and this one says why.
    #
    # Runs on its own connection from the `engine` fixture, not the shared
    # `connection` fixture: Postgres aborts the transaction on cancellation,
    # so reusing the suite-wide rolled-back transaction here would poison
    # every other test that runs afterwards. Rolling back and closing this
    # connection explicitly keeps the blast radius to this test alone.
    # (tests/test_notes_schema.py cites this test by name for that pattern.)
    conn = engine.connect()
    trans = conn.begin()
    try:
        conn.execute(text("SELECT set_config('statement_timeout', '100', true)"))
        with pytest.raises(OperationalError) as exc_info:
            conn.execute(text("SELECT pg_sleep(1)"))
        assert isinstance(exc_info.value.orig, psycopg.errors.QueryCanceled)
        assert exc_info.value.orig.sqlstate == "57014"
    finally:
        trans.rollback()
        conn.close()


def test_no_statement_timeout_is_in_force_before_any_search(connection):
    # A fresh transaction on which no search has run reads the server
    # default, so the cap exists only where a search set it: nothing in the
    # pool or the connection setup applies a timeout globally (ingest and the
    # startup schema checks rely on that).
    shown = connection.execute(text("SHOW statement_timeout")).scalar()
    assert shown == "0"
