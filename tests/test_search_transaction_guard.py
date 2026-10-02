"""The HNSW read-back guard in LearningRepository.search (#20).

search() sets hnsw.iterative_scan with set_config(..., true), which is SET LOCAL.
SET LOCAL only outlives its own statement when a transaction block is open. On
a connection with isolation_level="AUTOCOMMIT" there is none, so the setting
would vanish before the vector query ran and the search would silently fall
back to defaults -- reinstating #12 (short result sets, HTTP 200). search()
therefore reads the setting back and raises if it did not stick.

These tests deliberately do NOT use the `connection` / `db_session` fixtures.
Those wrap everything in an outer transaction plus a savepoint, so SET LOCAL
always persists there and the failure can never be reproduced through them.
Each test builds its own Session straight from the session-scoped `engine`.

Nothing here writes rows, so there is no data to clean up -- only the
session and connection to close.
"""
import pytest
from sqlalchemy.orm import Session

from app.repositories.learning import LearningRepository
from tests.fakes import fake_embedding


def test_search_raises_when_set_local_cannot_take_effect(engine):
    """Under AUTOCOMMIT, search() must fail loudly rather than run with defaults.

    Session.in_transaction() reports True here even though the database has no
    transaction block open, which is why the guard has to ask the database
    (current_setting) rather than the Session. Without the guard this call
    returns an (empty) list and the test fails -- that is the defect.
    """
    conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    session = Session(bind=conn)
    try:
        with pytest.raises(RuntimeError, match="did not take effect"):
            LearningRepository(session, "acme").search(
                fake_embedding("x"), top_k=5
            )
    finally:
        session.close()
        conn.close()


def test_search_does_not_raise_as_the_first_statement_of_a_normal_session(engine):
    """Positive control: the guard must not fire spuriously.

    search() is the very first statement on a fresh, ordinary Session. Before
    it runs, autobegin has not happened yet, so Session.in_transaction() is
    False -- a guard built on that check would have rejected this perfectly
    valid call. The read-back guard instead sees the transaction that the first
    execute() opens, and the setting persists across statements as intended.
    """
    session = Session(bind=engine)
    try:
        rows = LearningRepository(session, "acme").search(
            fake_embedding("x"), top_k=5
        )
        assert isinstance(rows, list)
    finally:
        session.rollback()
        session.close()
