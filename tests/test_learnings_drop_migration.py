"""Revision 3d67961c1596: drop the `learnings` table, reversibly.

The removed learnings API was never deployed, so the upgrade has no data to preserve. The
downgrade is still load-bearing: every revision below this one -- 9439c6362bb3
above all, whose own downgrade alters `learnings` -- expects to find the table,
so a downgrade that recreated it in any other shape would break the stack for
whoever next needs to unwind it. The table is recreated by raw DDL (the STORED
generated columns and the HNSW index have no SQLAlchemy spelling), and raw DDL
has no autogenerate, no model and no tests/test_schema_drift.py behind it. This
module is the only thing comparing it against the shape the historical
revisions actually produced.

That shape is pinned as a literal, copied from a pg_dump taken at 17be352b1a4c
before the drop revision existed. The catalog reads run with an empty
search_path, which makes PostgreSQL render types, defaults, generation
expressions and index definitions schema-qualified exactly as pg_dump does, so
the literals below are the dump's own text rather than a paraphrase of it.

Isolation follows tests/test_notes_schema.py::test_the_migration_is_reversible
and tests/test_passages_schema.py: each test takes its OWN connection, runs the
migrations inside one transaction and rolls it back in a `finally`. PostgreSQL
DDL is transactional and Alembic joins the transaction it is handed, so the
rollback undoes the CREATE/DROP TABLE, the recreated indexes and sequence, and
the `alembic_version` change; nothing leaks into the session-scoped database
the rest of the suite uses. None of these tests requests `connection` or
`db_session` -- a second open transaction would hold locks the DDL waits on.

Coverage across revisions: test_notes_schema's pinned downgrade to
9439c6362bb3 also crosses this revision (it runs this downgrade first, then
2eb0691f043c's and 17be352b1a4c's) and asserts `learnings` exists at the
bottom, so the round trip *through* this revision is covered there as well;
this module asserts what that test cannot, the exact recreated shape.

Migration targets are explicit revision ids, never `-1`: a relative target
silently changes meaning when a revision is added on top.
"""
from contextlib import contextmanager
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text

from alembic import command

# Resolved locally for the reason test_notes_schema.py gives: no dependency on
# a fixture module's private constants.
ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"

DROP_LEARNINGS = "3d67961c1596"  # the revision under test
BEFORE_DROP = "17be352b1a4c"  # the revision immediately below it

# (name, type, NOT NULL, attgenerated) in column order. attgenerated is 's'
# for a STORED generated column and '' otherwise. Order is part of the
# contract: it is the order the historical revisions appended columns in, and
# a positional INSERT written against the old table would put values in the
# wrong columns if it differed.
EXPECTED_COLUMNS = [
    ("id", "integer", True, ""),
    ("repo_name", "text", True, ""),
    ("file_path", "text", True, ""),
    ("content", "text", True, ""),
    ("metadata", "jsonb", True, ""),
    ("embedding", "public.vector(1024)", True, ""),
    ("client_name", "text", True, ""),
    ("created_at", "timestamp with time zone", True, ""),
    ("updated_at", "timestamp with time zone", True, ""),
    ("embedding_degenerate", "boolean", True, "s"),
    ("content_chars", "integer", True, "s"),
]

# Plain column defaults (pg_attrdef rows of non-generated columns).
EXPECTED_DEFAULTS = {
    "id": "nextval('public.learnings_id_seq'::regclass)",
    "created_at": "now()",
    "updated_at": "now()",
}

# STORED generation expressions (pg_attrdef rows of generated columns). These
# must match 9439c6362bb3's: the NaN self-distance is how a degenerate
# (zero-norm) vector is recognised, and a different expression would compute a
# different flag for the same row.
EXPECTED_GENERATED = {
    "embedding_degenerate": (
        "((embedding OPERATOR(public.<=>) embedding) = 'NaN'::double precision)"
    ),
    "content_chars": "char_length(content)",
}

# The complete index set. ix_learnings_file_path is deliberately absent:
# 3d80eb22a17d dropped it in favour of the composite index, so recreating it
# would produce a shape no historical revision ever had.
EXPECTED_INDEXES = {
    "learnings_pkey": (
        "CREATE UNIQUE INDEX learnings_pkey ON public.learnings USING btree (id)"
    ),
    "ix_learnings_client_repo_path": (
        "CREATE INDEX ix_learnings_client_repo_path ON public.learnings "
        "USING btree (client_name, repo_name, file_path)"
    ),
    "learnings_embedding_idx": (
        "CREATE INDEX learnings_embedding_idx ON public.learnings "
        "USING hnsw (embedding public.vector_cosine_ops)"
    ),
}


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


@contextmanager
def _pg_dump_rendering(conn):
    """Read the catalog with an empty search_path, as pg_dump does.

    Transaction-local, and restored on exit to the exact previous value, so
    a later Alembic command in the same transaction still finds
    `alembic_version` unqualified.
    """
    previous = conn.execute(text("SELECT current_setting('search_path')")).scalar_one()
    conn.execute(text("SELECT set_config('search_path', '', true)"))
    try:
        yield
    finally:
        conn.execute(
            text("SELECT set_config('search_path', :previous, true)"),
            {"previous": previous},
        )


def _revision(conn) -> str:
    return conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()


def _learnings_relations(conn) -> list[str]:
    """Every relation (table, index, sequence, view...) whose name mentions
    learnings, in any schema -- so an index or sequence orphaned by the drop
    is caught as well as the table itself."""
    return list(
        conn.execute(
            text(
                "SELECT relname FROM pg_class "
                "WHERE relname LIKE '%learnings%' ORDER BY relname"
            )
        ).scalars()
    )


def test_the_revision_under_test_sits_directly_on_the_passage_scope_revision():
    """"One step down" is a literal target, so pin what it is one step from.

    If a revision were ever inserted between these two, downgrading to
    BEFORE_DROP would also run that revision, and the shape assertions below
    would be measuring more than this migration.
    """
    script = ScriptDirectory.from_config(Config(str(ALEMBIC_INI)))

    assert script.get_revision(DROP_LEARNINGS).down_revision == BEFORE_DROP


def test_learnings_does_not_exist_at_head(engine):
    """The state the session fixture migrated to: no learnings, no leftovers."""
    with engine.connect() as conn:
        assert conn.execute(text("SELECT to_regclass('learnings')")).scalar() is None
        assert _learnings_relations(conn) == []


def test_the_downgrade_recreates_learnings_empty_and_in_its_historical_shape(engine):
    with _private_migration(engine) as (conn, alembic_config):
        command.downgrade(alembic_config, BEFORE_DROP)
        assert _revision(conn) == BEFORE_DROP

        assert conn.execute(
            text("SELECT to_regclass('learnings')")
        ).scalar() is not None
        # Recreated, not restored: there is nothing to restore rows from.
        assert conn.execute(text("SELECT count(*) FROM learnings")).scalar_one() == 0

        with _pg_dump_rendering(conn):
            columns = [
                (row.attname, row.type, row.attnotnull, row.attgenerated)
                for row in conn.execute(
                    text(
                        "SELECT attname, format_type(atttypid, atttypmod) AS type, "
                        "attnotnull, attgenerated::text AS attgenerated "
                        "FROM pg_attribute "
                        "WHERE attrelid = 'public.learnings'::regclass "
                        "AND attnum > 0 AND NOT attisdropped "
                        "ORDER BY attnum"
                    )
                )
            ]
            attrdefs = conn.execute(
                text(
                    "SELECT a.attname, a.attgenerated::text AS attgenerated, "
                    "pg_get_expr(d.adbin, d.adrelid) AS expr "
                    "FROM pg_attrdef d "
                    "JOIN pg_attribute a "
                    "ON a.attrelid = d.adrelid AND a.attnum = d.adnum "
                    "WHERE d.adrelid = 'public.learnings'::regclass"
                )
            ).all()
            indexes = dict(
                conn.execute(
                    text(
                        "SELECT indexname, indexdef FROM pg_indexes "
                        "WHERE schemaname = 'public' AND tablename = 'learnings'"
                    )
                ).all()
            )
            primary_key = conn.execute(
                text(
                    "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint "
                    "WHERE conrelid = 'public.learnings'::regclass AND contype = 'p'"
                )
            ).all()
            # serial's sequence must exist, be integer-typed (not bigint, which
            # `bigserial` or an identity column would give) and be OWNED BY
            # learnings.id so it is dropped with the table on the next upgrade.
            sequence = conn.execute(
                text(
                    "SELECT pg_get_serial_sequence('public.learnings', 'id'), "
                    "format_type(s.seqtypid, NULL) "
                    "FROM pg_sequence s "
                    "WHERE s.seqrelid = 'public.learnings_id_seq'::regclass"
                )
            ).one()

        assert columns == EXPECTED_COLUMNS
        assert {
            row.attname: row.expr for row in attrdefs if row.attgenerated == ""
        } == EXPECTED_DEFAULTS
        assert {
            row.attname: row.expr for row in attrdefs if row.attgenerated == "s"
        } == EXPECTED_GENERATED
        assert indexes == EXPECTED_INDEXES
        assert primary_key == [("learnings_pkey", "PRIMARY KEY (id)")]
        assert tuple(sequence) == ("public.learnings_id_seq", "integer")


def test_the_recreated_generated_columns_compute_what_9439c6362bb3s_did(engine):
    """The expressions are compared as text above; this proves they evaluate.

    A zero vector's cosine distance to itself is NaN in pgvector, which is the
    one input that must set embedding_degenerate; a unit vector must not. A
    non-ASCII character pins content_chars to characters, not bytes.
    """
    zero = "[" + ",".join(["0"] * 1024) + "]"
    unit = "[1" + ",0" * 1023 + "]"

    with _private_migration(engine) as (conn, alembic_config):
        command.downgrade(alembic_config, BEFORE_DROP)

        rows = conn.execute(
            text(
                "INSERT INTO learnings "
                "(repo_name, file_path, content, metadata, embedding, client_name) "
                "VALUES "
                "('repo_a', 'docs/zero.md', 'naïve', '{}', "
                "CAST(:zero AS vector), 'acme'), "
                "('repo_a', 'docs/unit.md', 'plain', '{}', "
                "CAST(:unit AS vector), 'acme') "
                "RETURNING file_path, embedding_degenerate, content_chars, "
                "id IS NOT NULL AS has_id, created_at IS NOT NULL AS has_created"
            ),
            {"zero": zero, "unit": unit},
        ).all()

        assert sorted(tuple(row) for row in rows) == [
            ("docs/unit.md", False, 5, True, True),
            ("docs/zero.md", True, 5, True, True),
        ]


def test_upgrading_back_to_head_drops_learnings_and_everything_it_owned(engine):
    with _private_migration(engine) as (conn, alembic_config):
        command.downgrade(alembic_config, BEFORE_DROP)
        # Non-empty on the way back up: the upgrade must drop a table that
        # holds rows, not merely an empty one.
        conn.execute(
            text(
                "INSERT INTO learnings "
                "(repo_name, file_path, content, metadata, embedding, client_name) "
                "VALUES ('repo_a', 'docs/a.md', 'body', '{}', "
                "CAST(:embedding AS vector), 'acme')"
            ),
            {"embedding": "[1" + ",0" * 1023 + "]"},
        )
        assert _learnings_relations(conn) != []

        command.upgrade(alembic_config, "head")

        head = ScriptDirectory.from_config(alembic_config).get_current_head()
        assert _revision(conn) == head
        assert conn.execute(text("SELECT to_regclass('learnings')")).scalar() is None
        assert conn.execute(text("SELECT to_regtype('learnings')")).scalar() is None
        # The pkey, both secondary indexes and the id sequence go with the table.
        assert _learnings_relations(conn) == []
        # The extension is not the drop's to remove: passages.embedding uses it.
        assert conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        ).scalar() == "vector"
