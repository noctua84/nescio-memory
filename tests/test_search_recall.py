"""Recall under tenant skew.

pgvector's HNSW index is approximate and post-filtered: it yields a bounded
candidate window and `WHERE client_name = ...` is applied afterwards. A client
holding a small share of the table can therefore have every candidate filtered
away and receive an empty result set with HTTP 200.

This test exercises the HNSW path deliberately. The rest of the suite forces
exact scans, which is why this defect stayed invisible.
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

    Index scans are re-enabled here deliberately: the `connection` fixture
    disables them suite-wide, so without this the HNSW path -- the one production
    uses -- would not be exercised at all.
    """
    db_session.execute(text("SET LOCAL enable_indexscan = on"))
    _seed_skewed_corpus(db_session)

    repo = LearningRepository(db_session, client_name="acme")
    rows = repo.search(fake_embedding("anything at all"), top_k=TOP_K)

    assert len(rows) == TOP_K, (
        f"expected {TOP_K} rows for a client with {SMALL_ROWS} of "
        f"{BULK_ROWS + SMALL_ROWS}, got {len(rows)} -- the HNSW candidate window "
        "was exhausted by other clients' rows before top_k matches were found"
    )
    assert {row[0].client_name for row in rows} == {"acme"}
