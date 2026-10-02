"""Recall under tenant skew.

pgvector's HNSW index is approximate and post-filtered: it yields a bounded
candidate window and `WHERE client_name = ...` is applied afterwards. A client
holding a small share of the table can therefore have every candidate filtered
away and receive an empty result set with HTTP 200.

This test exercises the HNSW path deliberately. The `connection` fixture forces
exact scans suite-wide so that ordering assertions elsewhere are deterministic,
so this test re-enables index scans for itself and then asserts the HNSW index is
genuinely used -- otherwise it would silently prove nothing.

The control assertion at the end of test_a_small_client_still_gets_the_full_top_k
verifies that the corpus still reproduces the HNSW starvation defect on the
current machine, ensuring the test meaningfully guards fix #12.
"""
from sqlalchemy import text

from app.models.learning import Learning
from app.repositories.learning import LearningRepository
from tests.factories import make_learning
from tests.fakes import fake_embedding

# BULK_ROWS is tuned so starvation is total on unfixed code: at this skew the
# unfixed query returned 0-1 rows of 10 across 20 query vectors. 4,000 did NOT
# reproduce the defect on the machine this was developed on. The control assertion
# at the end of test_a_small_client_still_gets_the_full_top_k enforces that this
# corpus still reproduces the defect on the current machine; if that assertion ever
# fails, raise BULK_ROWS and re-measure.
#
# SMALL_ROWS is deliberately LARGER than TOP_K. When it equalled TOP_K the query had
# to retrieve every row the client owns -- the hardest task the corpus can pose,
# because a single row landing outside HNSW's reachable set fails it outright. That
# made this test flake on 10-17% of container builds, returning 8 or 9. Asking for
# the nearest 10 of 40 lets a closer candidate substitute instead, which measured
# 12 of 12 builds returning exactly TOP_K.
BULK_ROWS = 8_000
SMALL_ROWS = 40
TOP_K = 10

# STARVATION_CEILING is the maximum number of rows the unfixed query can return
# for the control assertion. It's TOP_K // 2, leaving a margin to warn before the
# corpus drifts to the point of no longer reproducing the defect at all. Measured
# 0-1 of 10 at 8,000 rows.
STARVATION_CEILING = TOP_K // 2


def _seed_skewed_corpus(db_session) -> None:
    # The bulk rows are filler -- only their count and their presence in the
    # index matter -- so they are inserted in one statement rather than through
    # make_learning, which flushes per call. At these volumes that is the
    # difference between one round trip and several thousand.
    db_session.bulk_insert_mappings(
        Learning,
        [
            {
                "client_name": "bulk",
                "repo_name": "bulk_repo",
                "file_path": f"bulk/{index}.md",
                "content": f"bulk chunk {index}",
                "meta": {"chunk_index": index},
                "embedding": fake_embedding(f"bulk chunk {index}"),
            }
            for index in range(BULK_ROWS)
        ],
    )
    # The rows under test go through the factory, so they are built exactly as
    # the rest of the suite builds them.
    for index in range(SMALL_ROWS):
        make_learning(
            db_session,
            "acme",
            repo_name="acme_repo",
            file_path=f"acme/{index}.md",
            content=f"acme chunk {index}",
        )
    db_session.flush()


def test_a_small_client_still_gets_the_full_top_k(db_session):
    """A client holding a fraction of the table must still get top_k rows.

    The `connection` fixture forces exact scans suite-wide for determinism; this
    test opts back in to the HNSW path -- the one production uses -- and the plan
    assertion below verifies that opt-in actually took effect.

    The control assertion at the end verifies the corpus still starves the unfixed
    query, ensuring this test meaningfully guards fix #12.
    """
    # The connection fixture forces exact scans suite-wide for determinism. Opt back
    # in here: without this the HNSW path -- the one production takes -- would not be
    # exercised at all, which is how this defect stayed invisible. The plan assertion
    # below is what proves the opt-in actually worked.
    db_session.execute(text("SET LOCAL enable_indexscan = on"))
    _seed_skewed_corpus(db_session)

    repo = LearningRepository(db_session, client_name="acme")
    query_embedding = fake_embedding("anything at all")
    rows = repo.search(query_embedding, top_k=TOP_K)

    # Pin the plan. A count-only assertion passes just as happily when the
    # planner picks the btree on (client_name, repo_name, file_path) plus a
    # Sort, which is an EXACT search with perfect recall -- green while
    # exercising neither HNSW nor the fix. Projection width and pg_class
    # statistics both flip this choice, and autovacuum mutates pg_class mid-run,
    # so the index has to be asserted rather than assumed.
    vector_literal = "[" + ",".join(repr(c) for c in query_embedding) + "]"

    # Factor out the query for reuse in both EXPLAIN and the control assertion.
    search_query = (
        "SELECT learnings.*, 1 - (embedding <=> "
        f"'{vector_literal}') AS similarity FROM learnings "
        "WHERE client_name = 'acme' ORDER BY embedding <=> "
        f"'{vector_literal}' LIMIT {TOP_K}"
    )

    plan = "\n".join(
        row[0]
        for row in db_session.execute(text(f"EXPLAIN {search_query}")).all()
    )
    assert "learnings_embedding_idx" in plan, (
        "the search did not use the HNSW index, so this test is not exercising "
        f"the path production takes and proves nothing about #12. Plan:\n{plan}"
    )

    # Exact equality. An earlier revision asserted `>= TOP_K - 1`, because at
    # SMALL_ROWS == TOP_K this test genuinely flaked. Reshaping the corpus removed
    # the flake rather than tolerating it -- see SMALL_ROWS above for the measurement.
    # Spec D3 still holds in general: recall is improved, not guaranteed, and a client
    # whose every row is needed can still come up short. This corpus is shaped so that
    # case is not what is being asserted.
    assert len(rows) == TOP_K, (
        f"expected {TOP_K} rows for a client with {SMALL_ROWS} of "
        f"{BULK_ROWS + SMALL_ROWS}, got {len(rows)} -- the HNSW candidate window "
        "was exhausted by other clients' rows before top_k matches were found"
    )
    assert {row[0].client_name for row in rows} == {"acme"}

    # D1 chose strict_order over relaxed_order precisely so rows come back in
    # exact distance order; relaxed_order can emit them out of order. Nothing
    # else in the suite tests that on the HNSW path. This is independent of the
    # recall shortfall above: it checks ordering, not count.
    similarities = [row[1] for row in rows]
    assert similarities == sorted(similarities, reverse=True), (
        "strict_order must return rows in non-increasing similarity, got "
        f"{similarities}"
    )

    # Control assertion: verify the corpus still starves the unfixed HNSW query.
    # Disable iterative_scan to simulate the pre-#12 behaviour. If this assertion
    # fails, the corpus no longer reproduces the defect on this host -- raise BULK_ROWS
    # and re-measure.
    db_session.execute(text("SET LOCAL hnsw.iterative_scan = 'off'"))
    control_rows = db_session.execute(text(search_query)).all()

    assert len(control_rows) <= STARVATION_CEILING, (
        f"with iterative_scan='off' (the pre-#12 behaviour) the query returned "
        f"{len(control_rows)} of {TOP_K}, so BULK_ROWS={BULK_ROWS} no longer "
        f"starves the HNSW window on this host and this test no longer guards #12 -- "
        f"raise BULK_ROWS and re-measure (see issue #19)"
    )
