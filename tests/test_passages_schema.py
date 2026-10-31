"""Revision 17be352b1a4c: `passages.scope`, `embedding_degenerate` and the 3-column FK.

The revision copies the owning note's scope onto every passage so the scope
predicate is single-table on the table the HNSW index scans, makes the copy
undriftable by widening `fk_passages_note` to
`(note_id, client_name, scope) -> notes(id, client_name, scope)`, and adds a
STORED generated flag for vectors whose `<=>` against anything is NaN -- which
would otherwise score as a perfect match (#42/#53).

tests/test_schema_drift.py already proves those objects exist at head. What it
cannot prove is what they *do*, or that the revision moves existing data and
constraints correctly in both directions; that is this module.

Two isolation regimes, chosen per test:

* Tests that run migrations take their own connection and roll it back in a
  `finally` (`_private_migration`), exactly as
  tests/test_notes_schema.py::test_the_migration_is_reversible does and for
  the same reasons: PostgreSQL DDL is transactional and Alembic joins the
  transaction it is handed, so the rollback undoes every ALTER, every inserted
  row and the `alembic_version` change. They deliberately do not request
  `connection`/`db_session` -- a second open transaction holding a lock on
  `passages` would deadlock against the ALTER TABLE.
* Behavioural tests at head use the ordinary rolled-back fixtures.

Migration targets are explicit revision ids, never `-1`: a relative target
silently changes meaning when a revision is added on top.
"""
from contextlib import contextmanager
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from alembic import command
from app.config import settings
from tests.factories import make_api_key, make_note, make_passage

# Resolved locally for the reason test_notes_schema.py gives: no dependency on
# a fixture module's private constants.
ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"

PASSAGE_SCOPE = "17be352b1a4c"  # the revision under test
NOTES_AND_PASSAGES = "2eb0691f043c"  # the revision immediately below it


@contextmanager
def _private_migration(engine):
    """Yield (connection, alembic config) inside a transaction that is always
    rolled back, so every migration run here is invisible to the rest of the
    suite once the block exits."""
    alembic_config = Config(str(ALEMBIC_INI))
    conn = engine.connect()
    transaction = conn.begin()
    try:
        alembic_config.attributes["connection"] = conn
        yield conn, alembic_config
    finally:
        transaction.rollback()
        conn.close()


def _revision(conn) -> str:
    return conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()


def _passage_columns(conn) -> set[str]:
    return set(
        conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = 'passages'"
            )
        ).scalars()
    )


def _note_unique_constraints(conn) -> set[str]:
    return set(
        conn.execute(
            text(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = to_regclass('notes') AND contype = 'u'"
            )
        ).scalars()
    )


def _fk_shape(conn):
    """fk_passages_note as (local columns, referenced table, referenced columns,
    on-delete action), columns in key order.

    Read from pg_constraint rather than inspected by name alone, because the
    failure modes are a key that lost a column or an action that lost CASCADE --
    both of which leave a constraint of the right name in place.
    """
    return conn.execute(
        text(
            """
            SELECT
                array(SELECT a.attname::text
                      FROM unnest(c.conkey) WITH ORDINALITY AS k(attnum, i)
                      JOIN pg_attribute a
                        ON a.attrelid = c.conrelid AND a.attnum = k.attnum
                      ORDER BY k.i),
                c.confrelid::regclass::text,
                array(SELECT a.attname::text
                      FROM unnest(c.confkey) WITH ORDINALITY AS k(attnum, i)
                      JOIN pg_attribute a
                        ON a.attrelid = c.confrelid AND a.attnum = k.attnum
                      ORDER BY k.i),
                c.confdeltype::text
            FROM pg_constraint c
            WHERE c.conname = 'fk_passages_note'
              AND c.conrelid = to_regclass('passages')
            """
        )
    ).one_or_none()


def _column_flags(conn, column: str):
    """(NOT NULL, generated kind) for a passages column. attgenerated is 's'
    for a STORED generated column and '' for an ordinary one."""
    return tuple(
        conn.execute(
            text(
                "SELECT attnotnull, attgenerated::text FROM pg_attribute "
                "WHERE attrelid = to_regclass('passages') AND attname = :column "
                "AND NOT attisdropped"
            ),
            {"column": column},
        ).one()
    )


def _vector_literal(value: float) -> str:
    return "[" + ",".join([repr(value)] * settings.embedding_dimension) + "]"


def _unit_vector_literal() -> str:
    return "[" + ",".join(["1.0"] + ["0.0"] * (settings.embedding_dimension - 1)) + "]"


def _insert_note(conn, scope: str, path: str) -> int:
    return conn.execute(
        text(
            "INSERT INTO notes (client_name, scope, path_or_uri, content_hash, "
            "pipeline_version, body) "
            "VALUES ('acme', :scope, :path, 'deadbeef', 'v-test', 'a body') "
            "RETURNING id"
        ),
        {"scope": scope, "path": path},
    ).scalar_one()


def test_the_scope_revision_round_trips_its_columns_and_constraints(engine):
    """Downgrade to 2eb0691f043c restores the 2-column FK; upgrade restores all.

    The downgrade has an ordering hazard the upgrade does not: the 2-column FK
    can only be recreated once uq_notes_id_client exists again, and
    uq_notes_id_client_scope can only be dropped once nothing references it.
    Asserting the full shape on both sides is what shows each step ran, not
    merely that alembic stopped raising.
    """
    with _private_migration(engine) as (conn, alembic_config):
        command.downgrade(alembic_config, NOTES_AND_PASSAGES)

        assert _revision(conn) == NOTES_AND_PASSAGES
        columns = _passage_columns(conn)
        assert "scope" not in columns
        assert "embedding_degenerate" not in columns
        uniques = _note_unique_constraints(conn)
        assert "uq_notes_id_client" in uniques
        assert "uq_notes_id_client_scope" not in uniques
        assert _fk_shape(conn) == (
            ["note_id", "client_name"],
            "notes",
            ["id", "client_name"],
            "c",
        )

        command.upgrade(alembic_config, PASSAGE_SCOPE)

        assert _revision(conn) == PASSAGE_SCOPE
        assert {"scope", "embedding_degenerate"} <= _passage_columns(conn)
        uniques = _note_unique_constraints(conn)
        assert "uq_notes_id_client_scope" in uniques
        # Superseded by the wider key, and dropped so autogenerate and the
        # model (which no longer declares it) agree.
        assert "uq_notes_id_client" not in uniques
        assert _fk_shape(conn) == (
            ["note_id", "client_name", "scope"],
            "notes",
            ["id", "client_name", "scope"],
            "c",
        )
        # scope must end NOT NULL: a NULL scope would satisfy the FK vacuously
        # (MATCH SIMPLE skips rows with any NULL key column) and escape it.
        assert _column_flags(conn, "scope") == (True, "")
        # Generated by Postgres from the stored vector, never written by the app.
        assert _column_flags(conn, "embedding_degenerate") == (True, "s")


def test_the_upgrade_backfills_each_passage_with_its_own_notes_scope(engine):
    """Existing passages get the scope of the note they belong to.

    Two notes under different scopes, so a backfill that wrote a constant, or
    joined on something other than the note's identity, gives at least one of
    them the wrong value. The degenerate flag is computed for pre-existing rows
    too: the column is STORED, so the table rewrite fills it in.
    """
    with _private_migration(engine) as (conn, alembic_config):
        command.downgrade(alembic_config, NOTES_AND_PASSAGES)

        first = _insert_note(conn, "s1", "docs/first.md")
        second = _insert_note(conn, "s2", "docs/second.md")
        for note_id in (first, second):
            conn.execute(
                text(
                    "INSERT INTO passages "
                    "(note_id, client_name, ordinal, content, embedding, tsv) "
                    "VALUES (:note_id, 'acme', 0, 'c', CAST(:embedding AS vector), "
                    "to_tsvector('english', 'c'))"
                ),
                {"note_id": note_id, "embedding": _unit_vector_literal()},
            )

        command.upgrade(alembic_config, PASSAGE_SCOPE)

        backfilled = dict(
            conn.execute(
                text("SELECT note_id, scope FROM passages WHERE note_id IN (:a, :b)"),
                {"a": first, "b": second},
            ).all()
        )
        assert backfilled == {first: "s1", second: "s2"}
        flags = conn.execute(
            text(
                "SELECT embedding_degenerate FROM passages WHERE note_id IN (:a, :b)"
            ),
            {"a": first, "b": second},
        ).scalars().all()
        assert flags == [False, False]


@pytest.mark.parametrize(
    ("label", "embedding", "expected"),
    [
        ("all zero", _vector_literal(0.0), True),
        # 1e-40 is a float4 subnormal: non-zero as stored, but its squared norm
        # underflows to 0, so `<=>` is NaN exactly as for the zero vector. This
        # is the case Python-side validation cannot see (#42/#53).
        ("all subnormal", _vector_literal(1e-40), True),
        ("normal unit vector", _unit_vector_literal(), False),
    ],
)
def test_embedding_degenerate_reflects_the_stored_vector(
    connection, label, embedding, expected
):
    """The flag is derived from what is on disk, through raw SQL.

    Raw SQL rather than the factory so the vector reaches pgvector's own input
    parser unmodified -- the subnormal case is about what float4 storage does
    to the value, not what Python does to it.
    """
    note_id = _insert_note(connection, "repo_a", f"docs/{label}.md")
    connection.execute(
        text(
            "INSERT INTO passages "
            "(note_id, client_name, scope, ordinal, content, embedding, tsv) "
            "VALUES (:note_id, 'acme', 'repo_a', 0, 'c', CAST(:embedding AS vector), "
            "to_tsvector('english', 'c'))"
        ),
        {"note_id": note_id, "embedding": embedding},
    )

    degenerate, first_component = connection.execute(
        text(
            "SELECT embedding_degenerate, (embedding::real[])[1] "
            "FROM passages WHERE note_id = :note_id"
        ),
        {"note_id": note_id},
    ).one()

    if label == "all subnormal":
        # Guards the premise: had storage flushed the value to zero, this case
        # would silently duplicate the all-zero one.
        assert first_component > 0.0
    assert degenerate is expected, f"{label}: embedding_degenerate={degenerate}"


def test_a_passage_cannot_carry_a_scope_other_than_its_notes(db_session):
    """The 3-column FK rejects a passage whose scope disagrees with its note's.

    The mismatched scope is one the same tenant really uses on another note, so
    the only thing wrong is the (note, scope) pairing -- which is exactly what
    the old 2-column FK could not see: (note_id, client_name) matches.
    """
    note = make_note(db_session, "acme", scope="repo_a", path_or_uri="docs/a.md")
    make_note(db_session, "acme", scope="repo_b", path_or_uri="docs/b.md")

    with pytest.raises(IntegrityError) as exc_info:
        make_passage(db_session, note, scope="repo_b")

    assert "fk_passages_note" in str(exc_info.value)
    db_session.rollback()


def test_deleting_a_note_cascades_only_to_its_own_passages(db_session):
    """ON DELETE CASCADE survives the FK being rebuilt with a third column.

    The revision drops and recreates fk_passages_note, so the cascade is only
    as good as that create_foreign_key call. A sibling note in the same scope
    keeps its passages: the cascade follows the full key, not the scope.
    """
    doomed = make_note(db_session, "acme", scope="repo_b", path_or_uri="docs/x.md")
    kept = make_note(db_session, "acme", scope="repo_b", path_or_uri="docs/y.md")
    make_passage(db_session, doomed, ordinal=0)
    make_passage(db_session, doomed, ordinal=1)
    make_passage(db_session, kept, ordinal=0)

    db_session.execute(text("DELETE FROM notes WHERE id = :id"), {"id": doomed.id})

    counts = dict(
        db_session.execute(
            text(
                "SELECT note_id, count(*) FROM passages "
                "WHERE note_id IN (:doomed, :kept) GROUP BY note_id"
            ),
            {"doomed": doomed.id, "kept": kept.id},
        ).all()
    )
    assert counts == {kept.id: 1}


def test_ingest_stamps_each_passage_with_the_request_scope(client, db_session):
    """POST /api/v2/ingest writes passages.scope from the request's scope.

    The same path under two scopes, so a route that stamped a constant, or
    reused the first note's scope, gets one of the two wrong. The FK would turn
    that into a failed ingest (the status assertion); the read-back in SQL then
    pins the value actually stored to the request's scope.
    """
    key = make_api_key(db_session, "acme")
    note_ids = {}
    for scope in ("repo_a", "repo_b"):
        response = client.post(
            "/api/v2/ingest",
            json={
                "scope": scope,
                "path_or_uri": "docs/note.md",
                "content": "The marmoset throughput ledger, without headings.",
            },
            headers={"X-API-Key": key},
        )
        assert response.status_code == 200, response.text
        assert response.json()["passages"] >= 1
        note_ids[scope] = response.json()["note_id"]

    for scope, note_id in note_ids.items():
        stored = set(
            db_session.execute(
                text("SELECT scope FROM passages WHERE note_id = :id"),
                {"id": note_id},
            ).scalars()
        )
        assert stored == {scope}, f"note {note_id} ingested under {scope!r}"
