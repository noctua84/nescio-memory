"""Recall under tenant skew.

pgvector's HNSW index is approximate and post-filtered: it yields a bounded
candidate window and `WHERE client_name = ...` is applied afterwards. A client
holding a small share of the table can therefore have every candidate filtered
away and receive an empty result set with HTTP 200.

This test exercises the HNSW path deliberately. The `connection` fixture forces
exact scans suite-wide so that ordering assertions elsewhere are deterministic,
so this test re-enables index scans for itself and then asserts the HNSW index is
genuinely used -- otherwise it would silently prove nothing.

Seeding rebuilds the HNSW index from scratch instead of letting it grow
incrementally as the 8,000 filler rows are inserted. Measured: incremental HNSW
insertion of 8,000 384-dim vectors cost ~17s of this test's ~20s runtime, almost
entirely server-side (inside the DB call, not `fake_embedding` generation --
that's ~1s). Dropping the index, inserting plain rows, then building the index
once in bulk measured seed ~5s + build ~1.1s. See `_seed_skewed_corpus` and the
planner-pinning comment below for why that's safe and why it's mandatory.
"""
from sqlalchemy import text

from app.models.learning import Learning
from app.repositories.learning import LearningRepository
from tests.factories import make_learning
from tests.fakes import fake_embedding

# BULK_ROWS is tuned so starvation is total on unfixed code: at this skew the
# unfixed query returned 0-1 rows of 10 across 20 query vectors. 4,000 did NOT
# reproduce the defect on the machine this was developed on, so if this test ever
# passes BEFORE the fix, re-tune it rather than trusting it.
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


def _seed_skewed_corpus(db_session) -> None:
    # Drop the HNSW index before inserting and rebuild it once afterwards,
    # instead of letting pgvector grow it incrementally one row at a time.
    # Incremental insertion of 8,000 vectors measured ~17s server-side; a bulk
    # rebuild after the fact measured ~1.1s on top of a ~5s seed. The index
    # itself is unchanged -- only *when* it gets built.
    #
    # The definition is read from pg_indexes rather than hard-coded so it can
    # never drift from the migration that actually creates
    # learnings_embedding_idx (access method, operator class, any future
    # storage params). Asserting it was found turns a silent no-op (e.g. the
    # index renamed or dropped upstream) into a loud failure instead of this
    # test quietly reverting to the slow incremental path -- or worse, to no
    # index at all.
    #
    # This is safe against leaking into other tests because DDL in Postgres is
    # transactional: DROP/CREATE INDEX run inside the `connection` fixture's
    # outer transaction, which is always rolled back, so the real
    # learnings_embedding_idx is restored exactly as the migrations left it
    # once this test ends.
    indexdef = db_session.execute(
        text(
            "SELECT indexdef FROM pg_indexes "
            "WHERE indexname = 'learnings_embedding_idx'"
        )
    ).scalar()
    assert indexdef, (
        "learnings_embedding_idx not found in pg_indexes -- this test's bulk "
        "rebuild has nothing to rebuild from. Check the migration that creates "
        "it hasn't renamed or dropped the index."
    )
    db_session.execute(text("DROP INDEX learnings_embedding_idx"))

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

    # Rebuild the index now that every row is in place, from the exact DDL the
    # migration used -- same access method, same operator class, same (default)
    # storage params. This is the ~1.1s bulk build that replaces the ~17s of
    # incremental per-row insertion above.
    db_session.execute(text(indexdef))


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

    # Rebuilding the index (above) changes pg_class's statistics for
    # learnings_embedding_idx relative to incremental growth, and measured
    # behaviour was that this alone made the planner abandon HNSW for an exact
    # Index Scan + Sort on the (client_name, repo_name, file_path) btree. That
    # plan has perfect recall by construction, so the unfixed query -- no
    # set_config, no iterative_scan -- passed under it: the test would still go
    # green but would be proving nothing about #12. Disabling sort, bitmap and
    # seqscan plans forces the planner back onto the HNSW index scan that
    # production actually takes; the EXPLAIN assertion below is what confirms
    # that forcing worked rather than just trusting it.
    db_session.execute(text("SET LOCAL enable_sort = off"))
    db_session.execute(text("SET LOCAL enable_bitmapscan = off"))
    db_session.execute(text("SET LOCAL enable_seqscan = off"))

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
