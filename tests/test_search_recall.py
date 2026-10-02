"""Recall under tenant skew.

pgvector's HNSW index is approximate and post-filtered: it yields a bounded
candidate window and `WHERE client_name = ...` is applied afterwards. A client
holding a small share of the table can therefore have every candidate filtered
away and receive an empty result set with HTTP 200.

This test exercises the HNSW path deliberately. The `connection` fixture forces
exact scans suite-wide so that ordering assertions elsewhere are deterministic,
so this test re-enables index scans for itself and then asserts the HNSW index is
genuinely used -- otherwise it would silently prove nothing.
"""
from sqlalchemy import text

from app.models.learning import Learning
from app.repositories.learning import LearningRepository
from tests.factories import make_learning
from tests.fakes import fake_embedding

# Tuned so starvation is total rather than partial on unfixed code. At 1% skew
# the measured behaviour was 70 of 250 rows -- partial, so a test at that ratio
# could pass by luck. See the plan's Task 1 for the observed figures.
BULK_ROWS = 8_000
SMALL_ROWS = 10
TOP_K = 10


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
    plan = "\n".join(
        row[0]
        for row in db_session.execute(
            text(
                "EXPLAIN SELECT learnings.*, 1 - (embedding <=> "
                f"'{vector_literal}') AS similarity FROM learnings "
                "WHERE client_name = 'acme' ORDER BY embedding <=> "
                f"'{vector_literal}' LIMIT {TOP_K}"
            )
        ).all()
    )
    assert "learnings_embedding_idx" in plan, (
        "the search did not use the HNSW index, so this test is not exercising "
        f"the path production takes and proves nothing about #12. Plan:\n{plan}"
    )

    # Not `== TOP_K`. Spec D3: recall is improved, not guaranteed -- max_scan_tuples
    # bounds the work and HNSW's graph construction is randomized, so roughly one run
    # in fourteen legitimately returns TOP_K - 1. The defect this guards is total
    # starvation: pre-fix this same query returned 1 row of 10, which this still
    # catches. Tightening this to equality reintroduces a ~7% flake.
    assert len(rows) >= TOP_K - 1, (
        f"expected at least {TOP_K - 1} rows for a client with {SMALL_ROWS} of "
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
