"""What a restored database must prove before the heartbeat may be sent.

Every function takes a session on the restored database and raises
VerificationError when the check fails. The messages are safe for the journal:
client names and counts at most, never note content. tests/test_restore_check.py
runs these on a real restore on every run of the suite.
"""
from __future__ import annotations

import json

from sqlalchemy import text
from sqlalchemy.orm import Session

from scripts._backup_common import BackupError, Manifest, read_db_facts

HNSW_INDEX = "passages_embedding_idx"

# A vector identical to the probe's, in the same tenant but under another note,
# is an equally correct top hit: the same text ingested twice is ordinary. So the
# probe has to be unique within its client, or a healthy restore would fail.
# LIMIT 1 keeps it cheap, since the anti-join stops at the first usable passage.
_PROBE_SQL = text(
    """
    SELECT p.note_id, p.client_name, CAST(p.embedding AS text)
    FROM passages p
    WHERE NOT p.embedding_degenerate
      AND NOT EXISTS (
          SELECT 1 FROM passages other
          WHERE other.client_name = p.client_name
            AND other.note_id <> p.note_id
            AND other.embedding = p.embedding
      )
    ORDER BY p.id
    LIMIT 1
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


def assert_search_finds_a_stored_passage(db: Session) -> None:
    """A stored embedding, used as the query, must find its own note: through the
    search production runs, and then through the index itself.

    Needs no Ollama. The query text is empty, so the full-text arm contributes
    nothing and the vector arm alone decides. A restore has no planner statistics
    until something analyzes it, hence ANALYZE, so the search gets the plan a
    settled database would give it. That plan may rightly be an exact scan and
    sort on a small table or a small tenant, which would pass with a broken index.
    So the same question is then put straight to the index, with the planner forced
    onto it. search_guards' settings from the search are still in effect, since
    this is one transaction.

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

    probe = db.execute(_PROBE_SQL).one_or_none()
    if probe is None:
        raise VerificationError(
            "the restore holds no passage to search for: none exists with a usable "
            "embedding unique to its note"
        )
    note_id, client_name, stored = probe
    embedding = json.loads(stored)
    if len(embedding) != settings.embedding_dimension:
        raise VerificationError(
            f"EMBEDDING_DIMENSION is {settings.embedding_dimension} here but the "
            f"restored embeddings have {len(embedding)} dimensions"
        )

    db.execute(text("ANALYZE"))
    rows = PassageSearchRepository(db, client_name).search("", embedding, top_k=1)
    if not rows or rows[0].note_id != note_id:
        raise VerificationError(
            "searching with a stored passage's embedding did not return that "
            "passage's note as the top hit"
        )

    force_hnsw_plan(db)
    indexed = db.execute(
        _INDEX_PROBE_SQL, {"client_name": client_name, "embedding": stored}
    ).scalar_one_or_none()
    if indexed != note_id:
        raise VerificationError(
            f"the HNSW index {HNSW_INDEX} did not return a stored passage's note "
            "as the top hit for that passage's own embedding"
        )


def verify_restored_database(db: Session, manifest: Manifest) -> None:
    """Every assertion the heartbeat stands for. Raises on the first that fails."""
    assert_alembic_revision(db, manifest)
    assert_client_counts(db, manifest)
    assert_hnsw_index_valid(db)
    assert_search_finds_a_stored_passage(db)
