"""What a restored database must prove before the heartbeat may be sent.

Every function takes a session on the restored database and raises
VerificationError when the check fails. The messages are safe for the journal:
client names and counts at most, never note content. tests/test_restore_check.py
runs these on a real restore on every run of the suite.
"""
from __future__ import annotations

import json
import logging

from sqlalchemy import text
from sqlalchemy.orm import Session

from scripts._backup_common import BackupError, Manifest, read_db_facts

logger = logging.getLogger(__name__)

HNSW_INDEX = "passages_embedding_idx"

# Approximate nearest-neighbour search misses a stored vector's own note now and
# then, so a single probe would page the operator on a healthy restore. Many
# random probes turn that into a share: the check fails on a broken index, which
# misses most probes, and not on the odd miss, which is normal.
SEARCH_PROBE_COUNT = 100
MAX_MISS_SHARE = 0.25

# A random sample, drawing the ids first so that only the chosen rows have their
# embeddings read. Random on every run, so a run's probes are independent of the
# last run's. No uniqueness test per candidate: it would scan the whole tenant
# and compare every embedding, which does not scale. A vector shared with another
# note is handled after the search instead, by _TWIN_SQL.
_PROBE_SQL = text(
    """
    SELECT p.note_id, p.client_name, CAST(p.embedding AS text)
    FROM passages p
    WHERE p.id IN (
        SELECT id FROM passages WHERE NOT embedding_degenerate
        ORDER BY random() LIMIT :count
    )
    """
)

# Asked only when the top hit is another note than the probe's. If that note holds
# the identical vector the answer is equally correct, so it is not a miss. The
# cost is bounded by one note's passages, not by the tenant.
_TWIN_SQL = text(
    """
    SELECT EXISTS (
        SELECT 1 FROM passages
        WHERE client_name = :client_name AND note_id = :note_id
          AND embedding = CAST(:embedding AS vector)
    )
    """
)

_INDEX_PROBE_SQL = text(
    """
    SELECT note_id FROM passages
    WHERE client_name = :client_name AND NOT embedding_degenerate
    ORDER BY embedding <=> CAST(:embedding AS vector)
    LIMIT 1
    """
)


class VerificationError(BackupError):
    """The restored database does not say what the manifest says."""


def assert_alembic_revision(db: Session, manifest: Manifest) -> None:
    facts = read_db_facts(db.connection())
    if facts.alembic_revision != manifest.alembic_revision:
        raise VerificationError(
            f"alembic revision {facts.alembic_revision} does not match the "
            f"manifest's {manifest.alembic_revision}"
        )


def assert_client_counts(db: Session, manifest: Manifest) -> None:
    restored = read_db_facts(db.connection()).clients
    differing = sorted(
        client
        for client in restored.keys() | manifest.clients.keys()
        if restored.get(client) != manifest.clients.get(client)
    )
    if differing:
        raise VerificationError(
            f"note or passage counts differ from the manifest for clients: {differing}"
        )


def assert_hnsw_index_valid(db: Session) -> None:
    """The index must exist, be HNSW, and be valid: a restore that was killed, or
    that ran out of memory building it, leaves a table that answers queries
    correctly and slowly, which no count would notice."""
    valid = db.execute(
        text(
            "SELECT i.indisvalid FROM pg_index i "
            "JOIN pg_class c ON c.oid = i.indexrelid "
            "JOIN pg_class t ON t.oid = i.indrelid "
            "JOIN pg_am a ON a.oid = c.relam "
            "WHERE c.relname = :name AND t.relname = 'passages' AND a.amname = 'hnsw'"
        ),
        {"name": HNSW_INDEX},
    ).scalar_one_or_none()
    if valid is None:
        raise VerificationError(f"HNSW index {HNSW_INDEX} is missing from the restore")
    if not valid:
        raise VerificationError(f"HNSW index {HNSW_INDEX} is not valid (indisvalid is false)")


def force_hnsw_plan(db: Session) -> None:
    """For this transaction only, and only for _INDEX_PROBE_SQL. tests/
    test_restore_verification.py asserts the plan it gives.

    Not for the hybrid search: on its joins these settings cost a nested loop over
    every passage of the tenant, measured at 472 ms against 3.5 ms unforced on
    20,000 passages, which would grow into a false alarm at the 5 s statement
    timeout as the table does.
    """
    for planner_setting in ("enable_seqscan", "enable_bitmapscan", "enable_sort"):
        db.execute(text(f"SET LOCAL {planner_setting} = off"))


def _is_miss(
    db: Session, top_note_id: object, probe_note_id: int, client_name: str, stored: str
) -> bool:
    """False when the top hit is the probe's own note, or a note in the same client
    holding the identical vector. No top hit at all is a miss, with no query."""
    if top_note_id is None:
        return True
    if top_note_id == probe_note_id:
        return False
    return not db.execute(
        _TWIN_SQL,
        {"client_name": client_name, "note_id": top_note_id, "embedding": stored},
    ).scalar_one()


def assert_search_finds_a_stored_passage(db: Session) -> None:
    """Stored embeddings, used as the query, must find their own notes: through the
    search production runs, and then through the index itself.

    A random sample of SEARCH_PROBE_COUNT passages is probed in both arms, and
    each arm fails when the share that missed reaches MAX_MISS_SHARE. A hit counts
    when the top note is the probe's own, or holds the identical vector.

    Needs no Ollama. The query text is empty, so the full-text arm contributes
    nothing and the vector arm alone decides. A restore has no planner statistics
    until something analyzes it, hence ANALYZE, so the search gets the plan a
    settled database would give it. That plan may rightly be an exact scan and
    sort on a small table or a small tenant, which would pass with a broken index.
    So the same questions are then put straight to the index, with the planner
    forced onto it. search_guards' settings from the search are still in effect,
    since this is one transaction.

    The order is fixed: every production search first, then force_hnsw_plan once,
    then every index probe. The forced settings are SET LOCAL, so they last for the
    transaction, and the hybrid search under them is a nested loop over the tenant
    that would time out at scale. A production search must never run after
    force_hnsw_plan, so the two arms are not interleaved. Database errors are not
    caught: a statement timeout or an unreadable index must fail the check.

    Imported here because these modules build the API's settings on import, and
    a settings failure then surfaces inside restore_check.main()'s error handling.
    """
    from app.config import settings
    from app.repositories.passage_search import PassageSearchRepository

    if settings.vector_weight <= 0:
        raise VerificationError(
            "VECTOR_WEIGHT must be > 0 for the restore check: with the vector arm "
            "switched off the search cannot exercise the embeddings or the index"
        )

    rows = db.execute(_PROBE_SQL, {"count": SEARCH_PROBE_COUNT}).all()
    if not rows:
        raise VerificationError(
            "the restore holds no passage to search for: none exists with a usable "
            "embedding"
        )
    probes = [(note_id, client_name, stored) for note_id, client_name, stored in rows]
    first_embedding = json.loads(probes[0][2])
    if len(first_embedding) != settings.embedding_dimension:
        raise VerificationError(
            f"EMBEDDING_DIMENSION is {settings.embedding_dimension} here but the "
            f"restored embeddings have {len(first_embedding)} dimensions"
        )

    db.execute(text("ANALYZE"))
    total = len(probes)

    repositories: dict[str, PassageSearchRepository] = {}
    search_misses = 0
    for note_id, client_name, stored in probes:
        repository = repositories.setdefault(
            client_name, PassageSearchRepository(db, client_name)
        )
        hits = repository.search("", json.loads(stored), top_k=1)
        top = hits[0].note_id if hits else None
        if _is_miss(db, top, note_id, client_name, stored):
            search_misses += 1
    if search_misses / total >= MAX_MISS_SHARE:
        raise VerificationError(
            f"{search_misses} of {total} stored passages' embeddings did not return "
            "their own note as the top hit through the search "
            f"(fails at {MAX_MISS_SHARE:.0%})"
        )

    force_hnsw_plan(db)
    index_misses = 0
    for note_id, client_name, stored in probes:
        top = db.execute(
            _INDEX_PROBE_SQL, {"client_name": client_name, "embedding": stored}
        ).scalar_one_or_none()
        if _is_miss(db, top, note_id, client_name, stored):
            index_misses += 1
    if index_misses / total >= MAX_MISS_SHARE:
        raise VerificationError(
            f"the HNSW index {HNSW_INDEX} did not return their own note as the top "
            f"hit for {index_misses} of {total} stored passages' embeddings "
            f"(fails at {MAX_MISS_SHARE:.0%})"
        )

    logger.info(
        "restore search check passed: probes=%d search_misses=%d index_misses=%d",
        total,
        search_misses,
        index_misses,
    )


def verify_restored_database(db: Session, manifest: Manifest) -> None:
    """Every assertion the heartbeat stands for. Raises on the first that fails."""
    assert_alembic_revision(db, manifest)
    assert_client_counts(db, manifest)
    assert_hnsw_index_valid(db)
    assert_search_finds_a_stored_passage(db)
