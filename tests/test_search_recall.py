"""Recall under tenant skew, on the passage search production runs.

pgvector's HNSW index is approximate and post-filtered: it yields a bounded
candidate window and `WHERE client_name = ...` is applied afterwards. A client
holding a small share of the table can therefore have every candidate filtered
away and receive an empty result set with HTTP 200 (#12). The fix is
`hnsw.iterative_scan = strict_order`, applied per search by
app/repositories/search_guards.py, which keeps the index searching until enough
rows survive the filters.

This test exercises the HNSW path deliberately, through
PassageSearchRepository.search with the full-text arm switched off
(FTS_WEIGHT=0), so the vector arm alone decides the result. The `connection`
fixture forces exact scans suite-wide so that ordering assertions elsewhere are
deterministic; this test re-enables index scans for itself and asserts the HNSW
index is genuinely used -- otherwise it would silently prove nothing.

Seeding drops passages_embedding_idx, bulk-inserts plain rows and rebuilds the
index once, instead of letting it grow incrementally per inserted row. On the
v1 learnings table that measured ~17s incremental against ~5s seed + ~1.1s
build for 8,000 rows; see `_seed_skewed_corpus` and BULK_ROWS below.

The control assertion at the end of test_a_small_client_still_gets_the_full_top_k
runs the very same statement without iterative_scan and verifies the corpus
still starves it, ensuring the test meaningfully guards fix #12.
"""
from sqlalchemy import bindparam, select, text

from app.config import settings
from app.models.passage import Passage
from app.repositories.passage_search import PassageSearchRepository, hybrid_statement
from tests.factories import make_note, make_passage
from tests.fakes import fake_embedding
from tests.test_search_index_usage import arm_subtree, explain_json, summarise

# BULK_ROWS is tuned so starvation is near-total without iterative_scan.
# Re-measured for passages on 2026-10-31 (pgvector/pgvector:pg17, 1024-d
# fake_embedding vectors, SMALL_ROWS=40, vector arm at LIMIT TOP_K, 20 query
# vectors per size), rows returned of TOP_K=10:
#
#   BULK_ROWS   with iterative_scan   without (control)   seed time
#   2,000       10 on 20 of 20        0-2                  ~5s
#   4,000       10 on 20 of 20        0-1                  ~9s
#   6,000       10 on 20 of 20        0-2                  ~13s
#   8,000       10 on 20 of 20        0-2                  ~18s
#
# Unlike v1 learnings (where 4,000 did NOT reproduce the defect), passages
# starve from 2,000 up. 4,000 is chosen as the smallest size with headroom: it
# halves the seed time of the old 8,000 while the control stays well below
# STARVATION_CEILING. Seed time is dominated by generating and sending the
# vectors, not the index build. The control assertion at the end of
# test_a_small_client_still_gets_the_full_top_k enforces that this corpus still
# reproduces the defect on the current machine; if it ever fails, raise
# BULK_ROWS and re-measure.
#
# SMALL_ROWS is deliberately LARGER than TOP_K. When it equalled TOP_K the query had
# to retrieve every row the client owns -- the hardest task the corpus can pose,
# because a single row landing outside HNSW's reachable set fails it outright. That
# made the v1 version of this test flake on 10-17% of container builds, returning 8
# or 9. Asking for the nearest 10 of 40 lets a closer candidate substitute instead.
BULK_ROWS = 4_000
SMALL_ROWS = 40
TOP_K = 10

# STARVATION_CEILING is the maximum number of rows the query without iterative_scan
# may return for the control assertion. It's TOP_K // 2, leaving a margin to warn
# before the corpus drifts to the point of no longer reproducing the defect at all.
STARVATION_CEILING = TOP_K // 2

SMALL_CLIENT = "acme"
BULK_CLIENT = "bulk"
HNSW_INDEX = "passages_embedding_idx"


def _seed_skewed_corpus(db_session, bulk_rows: int = BULK_ROWS) -> None:
    """Seed the skewed corpus, bulk-rebuilding passages_embedding_idx and
    pinning the planner onto it."""
    # The definition is read from pg_indexes rather than hard-coded so it can
    # never drift from the migration that creates passages_embedding_idx.
    # Asserting it was found turns a silent no-op (the index renamed or
    # dropped upstream) into a loud failure. DDL in Postgres is transactional:
    # DROP/CREATE INDEX run inside the `connection` fixture's outer
    # transaction, which is always rolled back, so the real index is restored
    # exactly as the migrations left it once this test ends.
    indexdef = db_session.execute(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"),
        {"name": HNSW_INDEX},
    ).scalar()
    assert indexdef, (
        f"{HNSW_INDEX} not found in pg_indexes -- this test's bulk rebuild has "
        "nothing to rebuild from. Check the migration that creates it hasn't "
        "renamed or dropped the index."
    )
    db_session.execute(text(f"DROP INDEX {HNSW_INDEX}"))

    # The bulk rows are filler -- only their count and their presence in the
    # index matter -- so they go in as one executemany rather than through
    # make_passage, which flushes per call.
    bulk_note = make_note(
        db_session, BULK_CLIENT, scope="bulk_repo", path_or_uri="bulk/corpus.md"
    )
    db_session.execute(
        text(
            "INSERT INTO passages "
            "(note_id, client_name, scope, ordinal, content, embedding, tsv) "
            "VALUES (:note_id, :client_name, 'bulk_repo', :ordinal, :content, "
            "CAST(:embedding AS vector), to_tsvector('english', :content))"
        ),
        [
            {
                "note_id": bulk_note.id,
                "client_name": BULK_CLIENT,
                "ordinal": index,
                "content": f"bulk passage {index}",
                "embedding": str(fake_embedding(f"bulk passage {index}")),
            }
            for index in range(bulk_rows)
        ],
    )
    # The rows under test go through the factories, so they are built exactly
    # as the rest of the suite builds them.
    small_note = make_note(
        db_session, SMALL_CLIENT, scope="acme_repo", path_or_uri="acme/notes.md"
    )
    for index in range(SMALL_ROWS):
        make_passage(
            db_session, small_note, ordinal=index, content=f"acme passage {index}"
        )
    db_session.flush()

    db_session.execute(text(indexdef))
    # The connection fixture forces exact scans suite-wide for determinism.
    # Opt back in here: without this the HNSW path -- the one production takes
    # -- would not be exercised at all, which is how #12 stayed invisible.
    #
    # Rebuilding the index refreshes pg_class's row estimate, and on the removed learnings table that
    # alone made the planner abandon HNSW for an exact btree scan + Sort,
    # under which the query without iterative_scan has perfect recall by
    # construction: green, proving nothing about #12. Disabling sort, bitmap
    # and seqscan plans forces the planner onto the HNSW index scan; the plan
    # assertion in the test confirms the forcing worked rather than trusting it.
    db_session.execute(text("SET LOCAL enable_indexscan = on"))
    db_session.execute(text("SET LOCAL enable_sort = off"))
    db_session.execute(text("SET LOCAL enable_bitmapscan = off"))
    db_session.execute(text("SET LOCAL enable_seqscan = off"))


def _vector_only_statement(query_vector):
    """The exact statement PassageSearchRepository.search builds under the
    settings this test patches: vector arm only, at LIMIT TOP_K."""
    return hybrid_statement(
        SMALL_CLIENT, "anything at all", query_vector, TOP_K, None,
        candidates=settings.search_candidates,
        rrf_k=settings.rrf_k,
        fts_weight=settings.fts_weight,
        vector_weight=settings.vector_weight,
    )


def test_a_small_client_still_gets_the_full_top_k(db_session, monkeypatch):
    """A client holding a fraction of the table must still get top_k rows.

    The control assertion at the end verifies the corpus still starves the
    same statement without iterative_scan, ensuring this test meaningfully
    guards fix #12.
    """
    # FTS off: the vector arm alone decides the result. Candidates pinned to
    # TOP_K so the arm asks for the nearest 10 of SMALL_ROWS=40 -- the shape
    # SMALL_ROWS was chosen for -- rather than the default 50, which would be
    # more rows than the client owns.
    monkeypatch.setattr(settings, "fts_weight", 0.0)
    monkeypatch.setattr(settings, "vector_weight", 1.0)
    monkeypatch.setattr(settings, "search_candidates", TOP_K)
    _seed_skewed_corpus(db_session)

    query_vector = fake_embedding("anything at all")
    rows = PassageSearchRepository(db_session, SMALL_CLIENT).search(
        "anything at all", query_vector, TOP_K
    )

    # Pin the plan. A count-only assertion passes just as happily when the
    # planner picks an exact btree scan plus a Sort -- perfect recall, green
    # while exercising neither HNSW nor the fix. EXPLAIN runs the statement the
    # repository just ran, with the same bound Vector parameter.
    root = explain_json(db_session, _vector_only_statement(query_vector))
    vector_arm = arm_subtree(root, "<=>")
    assert any(
        node["Node Type"] == "Index Scan" and node.get("Index Name") == HNSW_INDEX
        for node in vector_arm
    ), (
        "the search did not use the HNSW index, so this test is not exercising "
        f"the path production takes and proves nothing about #12. Plan:\n"
        f"{summarise(root)}"
    )

    assert len(rows) == TOP_K, (
        f"expected {TOP_K} rows for a client with {SMALL_ROWS} of "
        f"{BULK_ROWS + SMALL_ROWS}, got {len(rows)} -- the HNSW candidate window "
        "was exhausted by other clients' rows before top_k matches were found"
    )
    returned_ids = [row.id for row in rows]
    owners = db_session.execute(
        select(Passage.client_name).where(Passage.id.in_(returned_ids))
    ).scalars().all()
    assert set(owners) == {SMALL_CLIENT}

    # Vector arm only: every row carries a vector rank and no FTS rank, and the
    # fused order is the vector rank order.
    assert [row.vector_rank for row in rows] == list(range(1, TOP_K + 1))
    assert all(row.fts_rank is None for row in rows)
    assert [row.score for row in rows] == sorted(
        (row.score for row in rows), reverse=True
    )

    # Rows come back in non-decreasing true cosine distance. Note this cannot
    # tell strict_order from relaxed_order: hybrid_statement re-ranks the arm's
    # rows with row_number() over (distance, id), so the final order is exact
    # whichever mode fed it. What it does catch is ranks not matching distance.
    qv = bindparam("qv", value=query_vector, type_=Passage.embedding.type)
    distances = dict(
        db_session.execute(
            select(Passage.id, Passage.embedding.cosine_distance(qv)).where(
                Passage.id.in_(returned_ids)
            )
        ).all()
    )
    ordered = [distances[row_id] for row_id in returned_ids]
    assert ordered == sorted(ordered), (
        f"rows must come back in non-decreasing cosine distance, got {ordered}"
    )

    # Control assertion: the same statement without iterative_scan, i.e. the
    # pre-#12 behaviour. SET LOCAL overrides the set_config(..., true) the
    # search applied, for the rest of this rolled-back transaction only. If this
    # fails, the corpus no longer reproduces the defect on this host -- raise
    # BULK_ROWS and re-measure.
    db_session.execute(text("SET LOCAL hnsw.iterative_scan = 'off'"))
    control_rows = db_session.execute(_vector_only_statement(query_vector)).all()

    assert len(control_rows) <= STARVATION_CEILING, (
        f"with iterative_scan='off' (the pre-#12 behaviour) the query returned "
        f"{len(control_rows)} of {TOP_K}, so BULK_ROWS={BULK_ROWS} no longer "
        f"starves the HNSW window on this host and this test no longer guards #12 -- "
        f"raise BULK_ROWS and re-measure (see issue #19)"
    )
