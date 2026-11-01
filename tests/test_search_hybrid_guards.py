"""The transaction-local guards every hybrid search runs under, and the
statement builder's refusal to build a search with no arm.

PassageSearchRepository.search applies hnsw.iterative_scan=strict_order,
hnsw.max_scan_tuples=100000 and statement_timeout (from settings) with
set_config(..., true) -- SET LOCAL -- and reads all three back, raising
RuntimeError("... did not take effect ...") if any did not stick.

The AUTOCOMMIT and leak tests deliberately avoid the db_session fixture: it
wraps everything in an outer transaction plus a savepoint, where SET LOCAL
always persists and neither failure can be reproduced. Nothing here writes
rows, so there is no data to clean up -- only sessions and connections.
"""
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

import app.config as config_module
from app.repositories.passage_search import PassageSearchRepository, hybrid_statement
from tests.fakes import unit_vector

_READ_BACK = text(
    "SELECT current_setting('hnsw.iterative_scan', true),"
    " current_setting('hnsw.max_scan_tuples', true),"
    " current_setting('statement_timeout')"
)


@pytest.mark.parametrize("query_vector", [None, unit_vector(0)],
                         ids=["fts-only", "hybrid"])
def test_search_applies_all_three_settings_to_its_transaction(
    db_session, monkeypatch, query_vector
):
    # 4321 is distinctive: neither the default (5000) nor Postgres's own (0).
    monkeypatch.setattr(config_module.settings, "statement_timeout_ms", 4321)

    PassageSearchRepository(db_session, "acme").search("pelicanium", query_vector, 5)

    shown = tuple(db_session.execute(_READ_BACK).one())
    assert shown == ("strict_order", "100000", "4321ms")


@pytest.mark.parametrize("milliseconds, rendered",
                         [(5000, "5s"), (60000, "1min"), (1, "1ms")])
def test_guard_accepts_postgres_normalised_timeout_spellings(
    db_session, monkeypatch, milliseconds, rendered
):
    # Postgres reports 5000 as "5s" and 60000 as "1min"; a guard comparing
    # against f"{ms}ms" alone would fire on a correctly applied setting.
    monkeypatch.setattr(config_module.settings, "statement_timeout_ms", milliseconds)

    PassageSearchRepository(db_session, "acme").search("pelicanium", None, 5)

    assert db_session.execute(text("SHOW statement_timeout")).scalar() == rendered


@pytest.mark.parametrize("query_vector", [None, unit_vector(0)],
                         ids=["fts-only", "hybrid"])
def test_read_back_guard_fires_without_an_open_transaction(engine, query_vector):
    # Under AUTOCOMMIT there is no transaction block, so set_config(..., true)
    # covers only its own statement and the search would run with defaults.
    connection = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    session = Session(bind=connection)
    try:
        with pytest.raises(RuntimeError, match="did not take effect"):
            PassageSearchRepository(session, "acme").search(
                "pelicanium", query_vector, 5)
    finally:
        session.close()
        connection.close()


def test_guards_do_not_outlive_the_search_transaction(engine, monkeypatch):
    # Positive control for the guard (first statement of an ordinary session
    # must not raise) and the reason it uses SET LOCAL: once the transaction
    # ends, the pooled connection is back to its own settings.
    monkeypatch.setattr(config_module.settings, "statement_timeout_ms", 4321)
    connection = engine.connect()
    session = Session(bind=connection)
    try:
        PassageSearchRepository(session, "acme").search("pelicanium", unit_vector(0), 5)
        assert tuple(session.execute(_READ_BACK).one()) == (
            "strict_order", "100000", "4321ms")
        session.rollback()

        iterative_scan, max_scan_tuples, timeout = session.execute(_READ_BACK).one()
        assert iterative_scan != "strict_order"
        assert max_scan_tuples != "100000"
        assert timeout != "4321ms"
    finally:
        session.rollback()
        session.close()
        connection.close()


_STATEMENT_ARGS = {"top_k": 5, "repo_filter": None, "candidates": 50, "rrf_k": 60}


@pytest.mark.parametrize(
    "query_vector, fts_weight, vector_weight",
    [
        (unit_vector(0), 0.0, 0.0),  # both weights off
        (None, 0.0, 1.0),            # vector arm has no vector, FTS off
        (None, 0.0, 0.0),
    ],
    ids=["both-weights-zero", "no-vector-fts-zero", "nothing-at-all"],
)
def test_hybrid_statement_with_no_arm_raises_value_error(
    query_vector, fts_weight, vector_weight
):
    with pytest.raises(ValueError, match="at least one arm"):
        hybrid_statement("acme", "pelicanium", query_vector,
                         fts_weight=fts_weight, vector_weight=vector_weight,
                         **_STATEMENT_ARGS)


@pytest.mark.parametrize(
    "query_vector, fts_weight, vector_weight",
    [(None, 1.0, 0.0), (None, 1.0, 1.0), (unit_vector(0), 0.0, 1.0)],
    ids=["fts-only", "fts-with-no-vector", "vector-only"],
)
def test_hybrid_statement_with_one_arm_builds(query_vector, fts_weight, vector_weight):
    # Control for the test above: one arm is enough, and building executes
    # nothing (no session is involved).
    stmt = hybrid_statement("acme", "pelicanium", query_vector,
                            fts_weight=fts_weight, vector_weight=vector_weight,
                            **_STATEMENT_ARGS)
    assert stmt is not None
