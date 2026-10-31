"""Proves the harness itself, then one end-to-end pass over the passages API.

If the harness checks fail, nothing else in the suite can be trusted. The
end-to-end test at the bottom is the shortest path through every /api/v2
route in the order a client would use them: ingest, search, get, delete.
"""
from sqlalchemy import inspect, select, text

from app.config import settings
from app.core.pipeline import content_hash, pipeline_version
from app.models.note import Note
from tests.factories import make_api_key


def test_migrations_created_every_table(engine):
    tables = set(inspect(engine).get_table_names())
    assert {"notes", "passages", "api_keys", "alembic_version"} <= tables


def test_the_vector_extension_is_installed(engine):
    with engine.connect() as connection:
        installed = connection.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
    assert installed == 1


def test_the_embedding_column_has_the_configured_dimension(engine):
    # passages.embedding is created as vector(1024) by 2eb0691f043c, the width
    # qwen3-embedding:0.6b actually emits (#36). Asserted against settings
    # rather than a literal so the column and the configuration cannot drift
    # apart silently -- but note that both can still be wrong about the real
    # model, which is what tests/test_embedding_width_contract.py covers.
    # If create_all() were ever substituted for real migrations this assertion
    # would still pass, which is why the HNSW check below exists too.
    with engine.connect() as connection:
        dimension = connection.execute(
            text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'passages'::regclass AND attname = 'embedding'"
            )
        ).scalar()
    assert dimension == settings.embedding_dimension


def test_the_hnsw_index_exists(engine):
    # Created only by migration 2eb0691f043c, as raw DDL. The model declares
    # it too (so autogenerate does not report it as removable), which means
    # create_all() would also produce it; the HNSW access method is checked
    # here as well so the assertion is about the real index, not its name.
    with engine.connect() as connection:
        definitions = dict(
            connection.execute(
                text(
                    "SELECT indexname, indexdef FROM pg_indexes "
                    "WHERE tablename = 'passages'"
                )
            ).all()
        )
    assert "passages_embedding_idx" in definitions
    assert "USING hnsw" in definitions["passages_embedding_idx"]
    assert "vector_cosine_ops" in definitions["passages_embedding_idx"]


def test_migrations_reached_the_alembic_head(engine):
    # Asserted against the script directory rather than a literal revision id,
    # so adding a migration does not break this test -- only failing to apply
    # one does.
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    from tests.conftest import ALEMBIC_INI

    heads = ScriptDirectory.from_config(Config(str(ALEMBIC_INI))).get_heads()
    with engine.connect() as connection:
        applied = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalars().all()
    assert sorted(applied) == sorted(heads)


def _probe_note(client_name: str) -> Note:
    body = "written by an isolation probe"
    return Note(
        client_name=client_name,
        scope="repo_a",
        path_or_uri="probe.md",
        body=body,
        content_hash=content_hash(body),
        pipeline_version=pipeline_version(),
        meta={},
    )


def test_a_row_written_in_a_test_is_visible_within_that_test(db_session):
    db_session.add(_probe_note("visibility_probe"))
    db_session.flush()
    found = db_session.scalars(
        select(Note).where(Note.client_name == "visibility_probe")
    ).all()
    assert len(found) == 1


def test_an_application_level_commit_does_not_escape_the_transaction(
    db_session, engine
):
    """The ingest and delete endpoints call db.commit() themselves.

    Under join_transaction_mode="create_savepoint" that releases a savepoint
    instead of committing the outer transaction. Proven here by reading through
    a second, independent connection while the test transaction is still open:
    the row must be visible to this test's own session and invisible to
    everyone else.

    Deliberately self-contained rather than split across two ordered tests. It
    cannot pass vacuously: the first assertion fails if the row was never
    written, so the second is only ever reached with a real row in play.
    """
    db_session.add(_probe_note("commit_probe"))
    db_session.commit()

    assert db_session.scalars(
        select(Note).where(Note.client_name == "commit_probe")
    ).all(), "the row is not visible to the session that wrote it"

    with engine.connect() as observer:
        escaped = observer.execute(
            select(Note.id).where(Note.client_name == "commit_probe")
        ).all()
    assert escaped == [], "a committed row escaped the test transaction"


# --- end-to-end over the passages API ----------------------------------------

SMOKE_NOTE = """---
name: Pelican throughput ledger
description: Synthetic note for the end-to-end smoke test.
---
# Overview

The pelican throughput ledger records quarterly widget counts.

# Method

Counts are reconciled against the marmoset audit every quarter.
"""


def test_ingest_search_get_delete_round_trip(client, db_session):
    key = make_api_key(db_session, "acme")
    headers = {"X-API-Key": key}

    # 1. Ingest.
    ingested = client.post(
        "/api/v2/ingest",
        json={
            "scope": "repo_a",
            "path_or_uri": "docs/pelican-ledger.md",
            "content": SMOKE_NOTE,
        },
        headers=headers,
    )
    assert ingested.status_code == 200, ingested.text
    ingest_body = ingested.json()
    assert ingest_body["skipped"] is False
    # >= 1, not one per heading: sections under MIN_SECTION_CHARS are merged
    # by the splitter (tests/test_splitter_short_sections.py), and splitting
    # is not what this smoke test is about.
    assert ingest_body["passages"] >= 1
    note_id = ingest_body["note_id"]

    # 2. Search finds a passage of that note.
    searched = client.post(
        "/api/v2/search",
        json={"q": "pelican throughput", "top_k": 5},
        headers=headers,
    )
    assert searched.status_code == 200, searched.text
    search_body = searched.json()
    assert search_body["degraded"] is False
    assert search_body["results"], "the ingested note was not found by search"
    top = search_body["results"][0]
    assert top["note_id"] == note_id
    assert top["path_or_uri"] == "docs/pelican-ledger.md"
    assert top["name"] == "Pelican throughput ledger"
    assert "pelican throughput ledger" in top["content"]
    # A real keyword match, not only a vector neighbour.
    assert top["fts_rank"] is not None

    # 3. Get returns the note as stored, with its passage count.
    fetched = client.get(f"/api/v2/notes/{note_id}", headers=headers)
    assert fetched.status_code == 200, fetched.text
    note = fetched.json()
    assert note["id"] == note_id
    assert note["scope"] == "repo_a"
    assert note["path_or_uri"] == "docs/pelican-ledger.md"
    assert note["name"] == "Pelican throughput ledger"
    assert note["description"] == "Synthetic note for the end-to-end smoke test."
    assert note["passage_count"] == ingest_body["passages"]
    assert note["content_hash"] == content_hash(SMOKE_NOTE)

    # 4. Delete, then the note and its passages are gone.
    deleted = client.delete(f"/api/v2/notes/{note_id}", headers=headers)
    assert deleted.status_code == 204
    assert deleted.content == b""

    gone = client.get(f"/api/v2/notes/{note_id}", headers=headers)
    assert gone.status_code == 404
    assert gone.json() == {"detail": "Note not found"}

    deleted_again = client.delete(f"/api/v2/notes/{note_id}", headers=headers)
    assert deleted_again.status_code == 404

    after = client.post(
        "/api/v2/search",
        json={"q": "pelican throughput", "top_k": 5},
        headers=headers,
    )
    assert after.status_code == 200
    assert after.json() == {"results": [], "degraded": False}
