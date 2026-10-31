"""POST /api/v2/search and PassageSearchRepository.search: the parts of the
hybrid contract tests/test_search_v2.py has no room for.

Split out of that module only to keep each under the length the plan asks for;
its docstring explains the `fts_only` fixture and why every route response is
asserted non-degraded. Here: a stop-word-only query, which the FTS arm cannot
answer and the vector arm does, and the transaction-local guards every search
runs under, exercised on the FTS-only path (query_vector=None) so they are
shown to cap a search that never embeds. The guards on the hybrid path are in
tests/test_search_hybrid_guards.py.
"""
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

import app.config as config_module
from app.repositories.passage_search import PassageSearchRepository
from tests.factories import make_api_key, make_note, make_passage


def _search(client, key, q, **overrides):
    payload = {"q": q, "top_k": 10}
    payload.update(overrides)
    return client.post("/api/v2/search", json=payload, headers={"X-API-Key": key})


def _results(response):
    assert response.status_code == 200, response.text
    assert response.json()["degraded"] is False
    return response.json()["results"]


def test_a_stop_word_only_query_is_answered_by_the_vector_arm(client, db_session):
    # The query test_an_empty_tsquery_returns_no_results_rather_than_an_error
    # sends with the vector arm off, now with it on: the FTS arm contributes
    # nothing, and the vector arm answers, so every row is a semantic-only match.
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme")
    passage = make_passage(
        db_session, note, content="The pelicanium isotope decays slowly."
    )

    results = _results(_search(client, key, "the"))

    assert [r["passage_id"] for r in results] == [passage.id]
    assert results[0]["fts_rank"] is None
    assert results[0]["vector_rank"] == 1
    assert results[0]["score"] > 0


def test_search_applies_the_configured_statement_timeout_to_its_transaction(
    db_session, monkeypatch
):
    # Modelled on the v1 test of the same name. A distinctive value, unlikely
    # to collide with either the default (5000) or the Postgres server default
    # (0), so this fails loudly if the set_config('statement_timeout', ...)
    # guard PassageSearchRepository.search applies is ever removed or stops
    # reading settings.statement_timeout_ms. query_vector=None: the guard
    # caps the FTS-only path too, not only hybrid searches.
    monkeypatch.setattr(config_module.settings, "statement_timeout_ms", 4321)

    PassageSearchRepository(db_session, "acme").search("pelicanium", None, 5)

    shown = db_session.execute(text("SHOW statement_timeout")).scalar()
    assert shown == "4321ms"


def test_the_read_back_guard_fires_without_an_open_transaction(engine):
    # Deliberately not the db_session fixture: that wraps everything in an
    # outer transaction plus a savepoint, where SET LOCAL always persists and
    # this failure cannot be reproduced. Under AUTOCOMMIT there is no
    # transaction block, so set_config(..., true) applies only to its own
    # statement and the search would otherwise run uncapped while still
    # answering 200.
    connection = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    session = Session(bind=connection)
    try:
        with pytest.raises(RuntimeError, match="did not take effect"):
            PassageSearchRepository(session, "acme").search(
                "pelicanium", query_vector=None, top_k=5
            )
    finally:
        session.close()
        connection.close()
