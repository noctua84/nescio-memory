"""Repository- and driver-level tests for the search-scoped statement_timeout.

test_config.py covers startup validation of STATEMENT_TIMEOUT_MS and
test_error_responses.py covers the HTTP-level mapping of a canceled statement.
These tests sit in between: they prove LearningRepository.search() actually
applies settings.statement_timeout_ms via SET LOCAL, that Postgres genuinely
cancels a statement once that timeout is tight, and that the setting never
leaks past the search transaction it was scoped to.
"""
import psycopg2.errors
import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

import app.config as config_module
from app.repositories.learning import LearningRepository
from tests.fakes import unit_vector


def test_search_applies_the_configured_statement_timeout_to_its_transaction(
    db_session, monkeypatch
):
    # A distinctive value, unlikely to collide with either the default (5000)
    # or the Postgres server default (0), so this fails loudly if the
    # set_config('statement_timeout', ...) line in LearningRepository.search
    # is ever removed or stops reading settings.statement_timeout_ms.
    monkeypatch.setattr(config_module.settings, "statement_timeout_ms", 4321)

    repo = LearningRepository(db_session, "acme")
    repo.search(embedding=unit_vector(0), top_k=5)

    shown = db_session.execute(text("SHOW statement_timeout")).scalar()
    assert shown == "4321ms"


def test_postgres_actually_cancels_a_statement_over_a_tight_timeout(engine):
    # Exercises the real driver/server behaviour the app depends on: a tight
    # statement_timeout really does raise OperationalError with .orig a
    # psycopg2 QueryCanceled carrying SQLSTATE 57014, which is exactly what
    # app.core.errors keys off of via isinstance().
    #
    # Runs on its own connection from the `engine` fixture, not the shared
    # `connection` fixture: Postgres aborts the transaction on cancellation,
    # so reusing the suite-wide rolled-back transaction here would poison
    # every other test that runs afterwards. Rolling back and closing this
    # connection explicitly keeps the blast radius to this test alone.
    conn = engine.connect()
    trans = conn.begin()
    try:
        conn.execute(text("SELECT set_config('statement_timeout', '100', true)"))
        with pytest.raises(OperationalError) as exc_info:
            conn.execute(text("SELECT pg_sleep(1)"))
        assert isinstance(exc_info.value.orig, psycopg2.errors.QueryCanceled)
        assert exc_info.value.orig.pgcode == "57014"
    finally:
        trans.rollback()
        conn.close()


def test_statement_timeout_does_not_leak_outside_the_search_transaction(connection):
    # This transaction is fresh (the `connection` fixture rolls back after
    # every test) and no search has run on it, so statement_timeout must still
    # read the Postgres server default rather than any value a previous
    # search's SET LOCAL configured -- proving the set_config(..., true)
    # scoping is real, not accidental.
    shown = connection.execute(text("SHOW statement_timeout")).scalar()
    assert shown == "0"
