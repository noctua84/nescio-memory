"""Recall under a selective repo_filter, on the passage search production runs.

tests/test_search_recall.py covers the tenant filter (client_name). repo_filter
is a second post-filter on the same HNSW index: `Passage.scope = :repo_filter`
is applied after the index yields its candidate window, so a scope holding a
small share of the table can have every candidate filtered away and return
nothing with HTTP 200, even for a client that owns the whole table. The same
`hnsw.iterative_scan = strict_order` (app/repositories/search_guards.py) is what
keeps the scan going until enough rows survive.

Here ONE client owns every row, so client_name filters nothing and the scope
alone is selective: SCOPED_ROWS passages in `target_repo` among BULK_ROWS in
`bulk_repo`. The full-text arm is off (FTS_WEIGHT=0) so the vector arm alone
decides the result; index scans are re-enabled and the plan is asserted to use
HNSW, since an exact scan has perfect recall by construction and would prove
nothing. The control run repeats the statement with iterative_scan off and must
starve, which is what ties this test to the guard.

Seeding follows test_search_recall.py: drop the HNSW index, bulk insert, rebuild
once. See that file for the BULK_ROWS measurements.
"""
from sqlalchemy import select, text

from app.config import settings
from app.models.passage import Passage
from app.repositories.passage_search import PassageSearchRepository, hybrid_statement
from tests.factories import make_note, make_passage
from tests.fakes import fake_embedding
from tests.test_search_index_usage import arm_subtree, explain_json, summarise
from tests.test_search_recall import HNSW_INDEX, TOP_K

CLIENT = "acme"
TARGET_SCOPE = "target_repo"
BULK_SCOPE = "bulk_repo"
# Measured 2026-11-01 (pgvector/pgvector:pg17, 1024-d fake_embedding vectors,
# SCOPED_ROWS=20, vector arm at LIMIT TOP_K, 20 query vectors per size), rows
# returned of TOP_K=10 with strict_order always 10 of 10 and the control
# (iterative_scan off) returning:
#
#   BULK_ROWS   control (min-max)
#   500         0-5   (reaches the ceiling, too close to use)
#   1,000       0-3
#   2,000       0-1
#
# 2,000 is the smallest size with a clear margin below STARVATION_CEILING; HNSW
# construction is randomised, so a size right at the edge would flake. If the
# control assertion ever fails, raise BULK_ROWS and re-measure.
BULK_ROWS = 2_000
SCOPED_ROWS = 20

# The control may return at most this many rows; half of TOP_K leaves a margin
# to warn before the corpus drifts to the point of not reproducing the defect.
STARVATION_CEILING = TOP_K // 2


def _seed(db_session) -> None:
    """One client, two scopes: a large bulk scope and a small target scope."""
    indexdef = db_session.execute(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"),
        {"name": HNSW_INDEX},
    ).scalar()
    assert indexdef, f"{HNSW_INDEX} not found in pg_indexes; nothing to rebuild from"
    # DDL is transactional: the index is restored when the fixture rolls back.
    db_session.execute(text(f"DROP INDEX {HNSW_INDEX}"))

    bulk_note = make_note(
        db_session, CLIENT, scope=BULK_SCOPE, path_or_uri="bulk/corpus.md"
    )
    db_session.execute(
        text(
            "INSERT INTO passages "
            "(note_id, client_name, scope, ordinal, content, embedding, tsv) "
            "VALUES (:note_id, :client_name, :scope, :ordinal, :content, "
            "CAST(:embedding AS vector), to_tsvector('english', :content))"
        ),
        [
            {
                "note_id": bulk_note.id,
                "client_name": CLIENT,
                "scope": BULK_SCOPE,
                "ordinal": index,
                "content": f"bulk passage {index}",
                "embedding": str(fake_embedding(f"bulk passage {index}")),
            }
            for index in range(BULK_ROWS)
        ],
    )
    target_note = make_note(
        db_session, CLIENT, scope=TARGET_SCOPE, path_or_uri="target/notes.md"
    )
    for index in range(SCOPED_ROWS):
        make_passage(
            db_session, target_note, ordinal=index, content=f"target passage {index}"
        )
    db_session.flush()

    db_session.execute(text(indexdef))
    # Same forcing as test_search_recall.py: without it the planner may pick an
    # exact scan after the rebuild and the test would pass for the wrong reason.
    db_session.execute(text("SET LOCAL enable_indexscan = on"))
    db_session.execute(text("SET LOCAL enable_sort = off"))
    db_session.execute(text("SET LOCAL enable_bitmapscan = off"))
    db_session.execute(text("SET LOCAL enable_seqscan = off"))


def _scoped_statement(query_vector):
    """The statement PassageSearchRepository.search builds with the vector arm
    only and repo_filter set."""
    return hybrid_statement(
        CLIENT, "anything at all", query_vector, TOP_K, TARGET_SCOPE,
        candidates=settings.search_candidates,
        rrf_k=settings.rrf_k,
        fts_weight=settings.fts_weight,
        vector_weight=settings.vector_weight,
    )


def test_a_selective_repo_filter_still_gets_the_full_top_k(db_session, monkeypatch):
    monkeypatch.setattr(settings, "fts_weight", 0.0)
    monkeypatch.setattr(settings, "vector_weight", 1.0)
    # The arm asks for the nearest TOP_K of SCOPED_ROWS, as in the tenant test.
    monkeypatch.setattr(settings, "search_candidates", TOP_K)
    _seed(db_session)

    query_vector = fake_embedding("anything at all")
    rows = PassageSearchRepository(db_session, CLIENT).search(
        "anything at all", query_vector, TOP_K, TARGET_SCOPE
    )

    root = explain_json(db_session, _scoped_statement(query_vector))
    vector_arm = arm_subtree(root, "<=>")
    assert any(
        node["Node Type"] == "Index Scan" and node.get("Index Name") == HNSW_INDEX
        for node in vector_arm
    ), (
        "the search did not use the HNSW index, so this test is not exercising "
        f"the path production takes. Plan:\n{summarise(root)}"
    )

    assert len(rows) == TOP_K, (
        f"expected {TOP_K} rows for a scope with {SCOPED_ROWS} of "
        f"{BULK_ROWS + SCOPED_ROWS} passages, got {len(rows)} -- the HNSW "
        "candidate window was exhausted by other scopes' rows"
    )
    scopes = db_session.execute(
        select(Passage.scope).where(Passage.id.in_([row.id for row in rows]))
    ).scalars().all()
    assert set(scopes) == {TARGET_SCOPE}
    assert all(row.fts_rank is None for row in rows)
    assert [row.vector_rank for row in rows] == list(range(1, TOP_K + 1))

    # Control: the same statement without iterative_scan must starve, or this
    # test no longer depends on the guard it exists to protect. SET LOCAL
    # overrides the search's set_config for the rest of this rolled-back
    # transaction only.
    db_session.execute(text("SET LOCAL hnsw.iterative_scan = 'off'"))
    control_rows = db_session.execute(_scoped_statement(query_vector)).all()
    assert len(control_rows) <= STARVATION_CEILING, (
        f"with iterative_scan='off' the query returned {len(control_rows)} of "
        f"{TOP_K}, so BULK_ROWS={BULK_ROWS} no longer starves the HNSW window "
        "for a selective scope on this host -- raise BULK_ROWS and re-measure"
    )
