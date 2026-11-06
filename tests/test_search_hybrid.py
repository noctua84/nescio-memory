"""PassageSearchRepository.search: the hybrid (FTS + vector, RRF-fused) contract.

Everything runs through the repository on the db_session fixture, where
conftest forces exact scans, so vector ranks below are exact arithmetic on
hand-built vectors rather than whatever an approximate index returns.

Every test pins the four search settings it depends on (`_pin`) instead of
trusting the defaults, so a changed default or a stray environment variable
cannot quietly change what is being asserted. Passage notes are created with
name=None and description=None so the tsv holds only the passage content and
FTS rank is decided by term frequency alone.
"""
import math

from sqlalchemy import text

import app.config as config_module
from app.config import settings
from app.repositories.passage_search import PassageSearchRepository
from tests.factories import make_note, make_passage
from tests.fakes import unit_vector

DIM = settings.embedding_dimension


def _pin(monkeypatch, **overrides):
    values = {"search_candidates": 50, "rrf_k": 60,
              "fts_weight": 1.0, "vector_weight": 1.0}
    values.update(overrides)
    for name, value in values.items():
        monkeypatch.setattr(config_module.settings, name, value)


def _vector(**axes: float) -> list[float]:
    """A unit vector with the given weights on axes named a0, a1, ..."""
    v = [0.0] * DIM
    for name, weight in axes.items():
        v[int(name[1:])] = weight
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _search(db, query, query_vector, top_k=10, repo_filter=None, client="acme"):
    return PassageSearchRepository(db, client).search(
        query, query_vector, top_k, repo_filter)


def _seed_three(db):
    """FTS order b (3 hits), c (2), a (1). Vector order against axis 0:
    a (distance 0), then b and c (both orthogonal, distance 1, id tiebreak).
    Fused at k=60: b = 1/61+1/62 > a = 1/63+1/61 > c = 1/62+1/63, an order
    that differs from both single-arm orders."""
    note = make_note(db, "acme")
    a = make_passage(db, note, ordinal=0, content="pelicanium alpha",
                     embedding=unit_vector(0))
    b = make_passage(db, note, ordinal=1, content="pelicanium pelicanium pelicanium beta",
                     embedding=unit_vector(1))
    c = make_passage(db, note, ordinal=2, content="pelicanium pelicanium gamma",
                     embedding=unit_vector(2))
    return a, b, c


def test_rrf_score_is_the_sum_of_one_over_k_plus_rank_per_arm(db_session, monkeypatch):
    _pin(monkeypatch)
    a, b, c = _seed_three(db_session)

    by_id = {r.id: r for r in _search(db_session, "pelicanium", unit_vector(0))}

    # b: ranked 1 by FTS and 2 by vector.
    assert (by_id[b.id].fts_rank, by_id[b.id].vector_rank) == (1, 2)
    assert math.isclose(by_id[b.id].score, 1 / 61 + 1 / 62, rel_tol=1e-12)
    assert (by_id[a.id].fts_rank, by_id[a.id].vector_rank) == (3, 1)
    assert math.isclose(by_id[a.id].score, 1 / 63 + 1 / 61, rel_tol=1e-12)
    assert (by_id[c.id].fts_rank, by_id[c.id].vector_rank) == (2, 3)
    assert math.isclose(by_id[c.id].score, 1 / 62 + 1 / 63, rel_tol=1e-12)
    assert all(isinstance(r.score, float) for r in by_id.values())


def test_weights_and_rrf_k_come_from_settings_at_call_time(db_session, monkeypatch):
    _pin(monkeypatch, rrf_k=10, fts_weight=2.0, vector_weight=0.5)
    _seed_three(db_session)

    rows = _search(db_session, "pelicanium", unit_vector(0))

    assert len(rows) == 3
    for r in rows:
        expected = 2.0 / (10 + r.fts_rank) + 0.5 / (10 + r.vector_rank)
        assert math.isclose(r.score, expected, rel_tol=1e-12)


def test_results_are_ordered_by_score_descending(db_session, monkeypatch):
    _pin(monkeypatch)
    a, b, c = _seed_three(db_session)

    rows = _search(db_session, "pelicanium", unit_vector(0))

    assert [r.id for r in rows] == [b.id, a.id, c.id]
    assert rows == sorted(rows, key=lambda r: (-r.score, r.id))


def test_ranks_are_none_outside_an_arms_candidates_and_ties_break_by_id(
    db_session, monkeypatch
):
    # n = max(2, top_k=2) = 2 candidates per arm.
    _pin(monkeypatch, search_candidates=2)
    note = make_note(db_session, "acme")
    # FTS ranks: a 1, b 2, d 3 (outside). Vector ranks against axis 0:
    # d 1, e 2, then a/b at distance 1 (outside).
    a = make_passage(db_session, note, ordinal=0,
                     content="pelicanium pelicanium pelicanium",
                     embedding=unit_vector(1))
    b = make_passage(db_session, note, ordinal=1, content="pelicanium pelicanium",
                     embedding=unit_vector(2))
    d = make_passage(db_session, note, ordinal=2, content="pelicanium",
                     embedding=unit_vector(0))
    e = make_passage(db_session, note, ordinal=3, content="unrelated words",
                     embedding=_vector(a0=1.0, a3=0.1))

    rows = _search(db_session, "pelicanium", unit_vector(0), top_k=2)

    # a and d each score exactly 1/61 from one arm; the tie goes to the lower id.
    assert [r.id for r in rows] == [a.id, d.id]
    assert (rows[0].fts_rank, rows[0].vector_rank) == (1, None)
    # d's text matches, but it is third by FTS and so outside that arm.
    assert (rows[1].fts_rank, rows[1].vector_rank) == (None, 1)
    assert rows[0].score == rows[1].score
    assert math.isclose(rows[0].score, 1 / 61, rel_tol=1e-12)
    assert b.id not in {r.id for r in rows} and e.id not in {r.id for r in rows}


def test_vector_weight_zero_drops_the_vector_arm(db_session, monkeypatch):
    _pin(monkeypatch, vector_weight=0.0)
    a, b, c = _seed_three(db_session)

    rows = _search(db_session, "pelicanium", unit_vector(0))

    assert [r.id for r in rows] == [b.id, c.id, a.id]  # the FTS order
    assert [r.fts_rank for r in rows] == [1, 2, 3]
    assert all(r.vector_rank is None for r in rows)
    for r in rows:
        assert math.isclose(r.score, 1 / (60 + r.fts_rank), rel_tol=1e-12)


def test_fts_weight_zero_drops_the_fts_arm(db_session, monkeypatch):
    _pin(monkeypatch, fts_weight=0.0)
    a, b, c = _seed_three(db_session)

    rows = _search(db_session, "pelicanium", unit_vector(0))

    assert [r.id for r in rows] == [a.id, b.id, c.id]  # the vector order
    assert [r.vector_rank for r in rows] == [1, 2, 3]
    assert all(r.fts_rank is None for r in rows)
    for r in rows:
        assert math.isclose(r.score, 1 / (60 + r.vector_rank), rel_tol=1e-12)


def test_no_query_vector_with_fts_weight_zero_runs_fts_only_with_positive_scores(
    db_session, monkeypatch
):
    _pin(monkeypatch, fts_weight=0.0)
    a, b, c = _seed_three(db_session)

    rows = _search(db_session, "pelicanium", None)

    assert [r.id for r in rows] == [b.id, c.id, a.id]
    assert [r.fts_rank for r in rows] == [1, 2, 3]
    assert all(r.vector_rank is None for r in rows)
    assert all(r.score > 0 for r in rows)
    # Weight 1.0 stands in for the zero FTS weight.
    for r in rows:
        assert math.isclose(r.score, 1 / (60 + r.fts_rank), rel_tol=1e-12)


def test_stop_word_only_query_still_returns_vector_rows(db_session, monkeypatch):
    _pin(monkeypatch)
    a, b, c = _seed_three(db_session)

    rows = _search(db_session, "the and of", unit_vector(0))

    assert [r.id for r in rows] == [a.id, b.id, c.id]
    assert all(r.fts_rank is None for r in rows)
    assert [r.vector_rank for r in rows] == [1, 2, 3]
    assert math.isclose(rows[0].score, 1 / 61, rel_tol=1e-12)


def _make_degenerate(db, note, ordinal, content, components):
    """A passage whose STORED embedding is degenerate, written with raw SQL so
    nothing on the Python side gets a chance to reject it."""
    passage = make_passage(db, note, ordinal=ordinal, content=content,
                           embedding=unit_vector(5))
    literal = "[" + ",".join(repr(x) for x in components) + "]"
    db.execute(text("UPDATE passages SET embedding = CAST(:e AS vector) WHERE id = :id"),
               {"e": literal, "id": passage.id})
    return passage


def test_degenerate_embedding_never_reaches_results_through_the_vector_arm(
    db_session, monkeypatch
):
    _pin(monkeypatch)
    note = make_note(db_session, "acme")
    good = make_passage(db_session, note, ordinal=0, content="ordinary text",
                        embedding=unit_vector(0))
    # All-subnormal in float4: against a real query its distance clamps to 0.0,
    # i.e. it would sit NEAREST to the query below if it were not excluded.
    subnormal = _make_degenerate(db_session, note, 1, "subnormal marmalade", [1e-40] * DIM)
    zero = _make_degenerate(db_session, note, 2, "zeroed marmalade", [0.0] * DIM)
    flags = dict(db_session.execute(
        text("SELECT id, embedding_degenerate FROM passages WHERE note_id = :n"),
        {"n": note.id}).all())
    assert flags == {good.id: False, subnormal.id: True, zero.id: True}

    query_vector = [1 / math.sqrt(DIM)] * DIM  # along the subnormal's direction

    rows = _search(db_session, "xyzzy", query_vector)
    assert [r.id for r in rows] == [good.id]

    # Their text is fine, so FTS may find them -- but never the vector arm.
    by_id = {r.id: r for r in _search(db_session, "marmalade", query_vector)}
    assert set(by_id) == {good.id, subnormal.id, zero.id}
    for degenerate in (subnormal, zero):
        assert by_id[degenerate.id].vector_rank is None
        assert by_id[degenerate.id].fts_rank is not None
    assert (by_id[good.id].fts_rank, by_id[good.id].vector_rank) == (None, 1)


def test_another_tenants_identical_passage_never_appears(db_session, monkeypatch):
    _pin(monkeypatch)
    mine_note = make_note(db_session, "acme")
    mine = make_passage(db_session, mine_note, content="ordinary words",
                        embedding=unit_vector(1))
    theirs_note = make_note(db_session, "globex")
    theirs = make_passage(db_session, theirs_note, content="pelicanium",
                          embedding=unit_vector(0))

    for query in ("pelicanium", "ordinary"):
        rows = _search(db_session, query, unit_vector(0))
        assert theirs.id not in {r.id for r in rows}
        assert [r.id for r in rows] == [mine.id]
    assert [r.id for r in _search(db_session, "pelicanium", None)] == []


def test_selective_repo_filter_returns_exactly_the_matching_passages(
    db_session, monkeypatch
):
    _pin(monkeypatch)
    bulk_note = make_note(db_session, "acme", scope="repo_bulk", path_or_uri="bulk.md")
    target_note = make_note(db_session, "acme", scope="repo_target",
                            path_or_uri="target.md")
    # 197 bulk passages beat every target passage on BOTH arms: three query
    # hits each and a vector equal to the query. Filtering only after the
    # arms' LIMIT would leave nothing of the target scope to return. The tsv
    # carries weight B, as make_passage's content does: unweighted (D), three hits
    # rank 0.3 against a target's 0.4, and whether a target then reached the top
    # results hung on which of the tied vector rows the scan happened to keep.
    exact = "[" + ",".join(repr(x) for x in unit_vector(0)) + "]"
    db_session.execute(
        text("INSERT INTO passages (note_id, client_name, scope, ordinal, content,"
             " embedding, tsv) VALUES (:n, 'acme', 'repo_bulk', :o, :t,"
             " CAST(:e AS vector), setweight(to_tsvector('english', :t), 'B'))"),
        [{"n": bulk_note.id, "o": i, "e": exact,
          "t": f"pelicanium pelicanium pelicanium bulk {i}"} for i in range(197)],
    )
    targets = [
        make_passage(db_session, target_note, ordinal=i, content=f"pelicanium target {i}",
                     embedding=unit_vector(1 + i))
        for i in range(3)
    ]
    total = db_session.execute(
        text("SELECT count(*) FROM passages WHERE client_name = 'acme'")).scalar()
    assert total == 200

    unfiltered = _search(db_session, "pelicanium", unit_vector(0))
    assert not {t.id for t in targets} & {r.id for r in unfiltered}

    rows = _search(db_session, "pelicanium", unit_vector(0), repo_filter="repo_target")
    assert sorted(r.id for r in rows) == sorted(t.id for t in targets)
    assert all(r.scope == "repo_target" for r in rows)


def test_each_arm_takes_max_of_candidates_and_top_k(db_session, monkeypatch):
    _pin(monkeypatch, search_candidates=2)
    note = make_note(db_session, "acme")
    for i in range(7):
        make_passage(db_session, note, ordinal=i, content=f"pelicanium item {i}",
                     embedding=unit_vector(i))

    assert len(_search(db_session, "pelicanium", unit_vector(0), top_k=5)) == 5
    # Single-arm runs too: neither arm alone may be capped at 2.
    assert len(_search(db_session, "pelicanium", None, top_k=5)) == 5
    assert len(_search(db_session, "the and of", unit_vector(0), top_k=5)) == 5
