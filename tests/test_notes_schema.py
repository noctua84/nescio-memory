"""Constraint-level guards for the notes/passages schema.

`tests/test_schema_drift.py` already compares `app.models.Base.metadata` against
the migrated database generically, with `ACCEPTED_DRIFT == set()`, so every
column, index and constraint the model declares is checked for existence by that
module and needs no restating here. What autogenerate cannot express is what the
constraints *do*: a `UNIQUE` that is declared but references the wrong columns,
an `ON DELETE CASCADE` that was written `NO ACTION`, a composite FK that is
really two single-column FKs, a GIN index that exists but cannot serve the query
it was created for. Those are behavioural facts and they are what this module
asserts, by provoking each constraint through a real PostgreSQL server.

The most important of them is the composite FK
`(note_id, client_name) -> notes(id, client_name)`. Every v2 query filters
`passages.client_name` as a single-table predicate rather than joining through
`notes` (see app/repositories/note.py's module docstring), which is correct only
while that column cannot disagree with the note's. The FK is the only thing that
makes it so; `test_a_passage_cannot_name_another_tenants_note` is the only thing
that makes the FK load-bearing in CI.

Every test here writes through the `db_session`/`connection` fixtures, whose
outer transaction is rolled back, except `test_the_migration_is_reversible`,
which needs DDL on the shared schema and therefore takes its own connection --
see that test for why that cannot disturb the rest of the run.
"""
import pytest
from alembic import command
from alembic.config import Config
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.config import settings
from tests.factories import make_note, make_passage

# Resolved here rather than imported from tests/conftest.py: this module must
# not depend on a fixture module's private constants, and the path is the same
# assumption alembic.ini's `prepend_sys_path = "."` already makes.
ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"


def _passage_count(db_session, note_id: int) -> int:
    """Count a note's passages in SQL, bypassing the ORM identity map.

    `db_session.get(Passage, ...)` would answer from the identity map and report
    a cascaded-away row as still present, which is precisely the failure
    `test_deleting_a_note_cascades_to_its_passages` exists to catch.
    """
    return db_session.execute(
        text("SELECT count(*) FROM passages WHERE note_id = :note_id"),
        {"note_id": note_id},
    ).scalar_one()


def test_the_notes_identity_is_unique_per_client_scope_and_path(db_session):
    """uq_notes_client_scope_path is the identity ingest upserts on.

    If it stopped covering all three columns, POST /api/v2/ingest's
    `ON CONFLICT (constraint="uq_notes_client_scope_path")` would either fail to
    match or match the wrong row, and the same note could exist twice.
    """
    make_note(db_session, "acme", scope="repo_a", path_or_uri="docs/note.md")

    with pytest.raises(IntegrityError) as exc_info:
        make_note(db_session, "acme", scope="repo_a", path_or_uri="docs/note.md")

    assert "uq_notes_client_scope_path" in str(exc_info.value)
    # Back to the savepoint: the failed flush left this session in an aborted
    # state, and without this the fixture's outer rollback is the only thing
    # that still works.
    db_session.rollback()


def test_the_same_identity_coexists_under_two_clients(db_session):
    """The other half of the identity: client_name is part of it.

    Two tenants ingesting the same path under the same scope is the normal
    case, not a conflict. A unique key of (scope, path_or_uri) alone would make
    the second tenant's ingest fail -- and a test that only asserts the
    duplicate is rejected passes happily against that broken key.
    """
    first = make_note(db_session, "acme", scope="repo_a", path_or_uri="docs/note.md")
    second = make_note(db_session, "other", scope="repo_a", path_or_uri="docs/note.md")

    assert first.id != second.id
    rows = db_session.execute(
        text(
            "SELECT client_name FROM notes "
            "WHERE scope = 'repo_a' AND path_or_uri = 'docs/note.md' "
            "ORDER BY client_name"
        )
    ).scalars().all()
    assert rows == ["acme", "other"]


def test_scope_defaults_to_the_empty_string_at_the_database_level(db_session):
    """`scope` must default to '' in the database, not to NULL.

    NULLs are not equal to each other in a unique key, so a NULL scope would let
    the same (client, path) be inserted without limit -- defeating
    uq_notes_client_scope_path entirely. The default is therefore asserted
    through raw SQL that omits the column, which is the only way to observe the
    server-side default: the ORM model would supply its own value and the
    factory always passes one.
    """
    db_session.execute(
        text(
            "INSERT INTO notes (client_name, path_or_uri, content_hash, "
            "pipeline_version, body) "
            "VALUES ('acme', 'docs/defaulted.md', 'deadbeef', 'v-test', 'a body')"
        )
    )

    scope = db_session.execute(
        text(
            "SELECT scope FROM notes "
            "WHERE client_name = 'acme' AND path_or_uri = 'docs/defaulted.md'"
        )
    ).scalar_one()

    assert scope == "", f"expected the empty string, got {scope!r}"


def test_a_passage_cannot_name_another_tenants_note(db_session):
    """The single most important constraint in the schema.

    `passages.client_name` is duplicated from the note so the tenant predicate
    stays single-table on the scanned table (issues #12/#74). That duplication is
    safe only because the composite FK (note_id, client_name) ->
    notes(id, client_name) makes a disagreeing pair unrepresentable. Two
    independent single-column FKs would accept this insert: note_id 1 exists and
    the client_name column has no FK of its own.
    """
    note = make_note(db_session, "acme")
    make_note(db_session, "other", path_or_uri="docs/other.md")

    with pytest.raises(IntegrityError) as exc_info:
        make_passage(db_session, note, client_name="other")

    assert "fk_passages_note" in str(exc_info.value)
    db_session.rollback()


def test_deleting_a_note_cascades_to_its_passages(db_session):
    """ON DELETE CASCADE on fk_passages_note, asserted at the schema level.

    DELETE /api/v2/notes/{id} deletes the note row only and relies on the
    cascade for its passages (NoteRepository.delete_note documents that choice).
    Without the cascade that route would either fail on the FK or orphan every
    passage, and a route-level test could only observe the 204 either way.
    """
    note = make_note(db_session, "acme")
    make_passage(db_session, note, ordinal=0)
    make_passage(db_session, note, ordinal=1)
    assert _passage_count(db_session, note.id) == 2

    db_session.execute(
        text("DELETE FROM notes WHERE id = :note_id"), {"note_id": note.id}
    )

    assert _passage_count(db_session, note.id) == 0


def test_passage_ordinals_are_unique_within_a_note(db_session):
    """uq_passages_note_ordinal: a note cannot have two passages at one ordinal.

    replace_passages deletes and reinserts wholesale, so a duplicate ordinal
    means the delete did not cover everything it should have -- which this
    constraint turns into a failed ingest rather than a silently doubled note.
    """
    note = make_note(db_session, "acme")
    make_passage(db_session, note, ordinal=0)

    with pytest.raises(IntegrityError) as exc_info:
        make_passage(db_session, note, ordinal=0, content="a different passage")

    assert "uq_passages_note_ordinal" in str(exc_info.value)
    db_session.rollback()


def test_the_same_ordinal_is_allowed_under_a_different_note(db_session):
    """The scope of that uniqueness: per note, not per tenant.

    Every note starts its passages at ordinal 0, so a unique key on
    (client_name, ordinal) -- or on ordinal alone -- would let a tenant store
    exactly one note's worth of passages. The rejection test above cannot tell
    the two apart.
    """
    first = make_note(db_session, "acme", path_or_uri="docs/first.md")
    second = make_note(db_session, "acme", path_or_uri="docs/second.md")

    make_passage(db_session, first, ordinal=0)
    make_passage(db_session, second, ordinal=0)

    assert _passage_count(db_session, first.id) == 1
    assert _passage_count(db_session, second.id) == 1


def test_the_passages_embedding_column_has_the_configured_width(engine):
    """The three-way embedding-width contract, on the new table.

    model <-> EMBEDDING_DIMENSION <-> vector(N). The migration hard-codes
    vector(1024) while app/models/passage.py declares
    Vector(settings.embedding_dimension), so the two agree only while the
    setting is 1024 -- and nothing else in the suite notices the difference,
    because tests.fakes.fake_embedding produces settings.embedding_dimension
    floats by construction and would match either way.

    The query mirrors app/core/schema_checks.py's own: pgvector stores the
    declared dimension directly in atttypmod, so the startup check and this test
    read the same number the same way. Issue #36 was exactly this disagreement on
    `learnings`, surviving 93 passing tests.
    """
    with engine.connect() as connection:
        atttypmod = connection.execute(
            text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = to_regclass('passages') AND attname = 'embedding'"
            )
        ).scalar()

    assert atttypmod == settings.embedding_dimension, (
        f"passages.embedding is vector({atttypmod}) but EMBEDDING_DIMENSION is "
        f"{settings.embedding_dimension}. Every v2 ingest would fail at insert "
        "time and verify_embedding_dimension would refuse to boot. Either "
        "correct EMBEDDING_DIMENSION or migrate the column."
    )


def test_the_tsv_gin_index_is_used_for_a_search(db_session):
    """ix_passages_tsv must be *usable* for the FTS predicate search_fts issues.

    A GIN index that exists satisfies test_schema_drift.py while being unusable
    for the real query -- a wrong opclass, or an index on an expression rather
    than the column, would both pass there. EXPLAIN is the only way to tell.

    This asserts usability, not planner preference, and the test is written
    around two measured facts about this scale.

    First, the test corpus is a handful of rows inside a rolled-back
    transaction, so pg_class's estimate for `passages` is ~0 and a sequential
    scan is genuinely the cheaper plan. `SET LOCAL enable_seqscan = off` takes
    that option away -- scoped to the transaction the `connection` fixture rolls
    back -- so the planner has to show whether it *can* use an index at all. The
    fixture also sets `enable_indexscan = off` suite-wide and leaves bitmap scans
    available, which is the only shape GIN supports anyway.

    Second, the EXPLAINed query carries the FTS predicate alone, without
    search_fts's `client_name = ...`. With the tenant predicate present the
    planner picks `Bitmap Index Scan on ix_passages_client_note` and demotes the
    tsquery to a heap `Filter` (measured; a correct and, at two rows, cheaper
    plan -- and the very reason that index exists), which would make this test
    fail while the GIN index is perfectly healthy. Dropping the predicate leaves
    the GIN index as the only index that can serve the query, so the assertion
    answers the question actually being asked: can `tsv @@ websearch_to_tsquery`
    be answered from ix_passages_tsv. What the planner prefers once the corpus is
    big enough is not a schema fact and is not asserted anywhere here.
    """
    note = make_note(
        db_session, "acme", name="A note about hydrostatics", description="a summary"
    )
    make_passage(db_session, note, content="the passage mentions hydrostatics plainly")

    db_session.execute(text("SET LOCAL enable_seqscan = off"))

    # The FTS predicate from NoteRepository.search_fts verbatim, including
    # websearch_to_tsquery and the 'english' configuration -- an index built
    # under a different configuration would not serve this.
    plan = "\n".join(
        row[0]
        for row in db_session.execute(
            text(
                "EXPLAIN SELECT id FROM passages "
                "WHERE tsv @@ websearch_to_tsquery('english', 'hydrostatics')"
            )
        ).all()
    )

    assert "ix_passages_tsv" in plan, (
        "the FTS predicate cannot be served by the GIN index on passages.tsv, "
        "so every v2 search is a sequential scan over the whole tenant corpus. "
        f"Plan:\n{plan}"
    )
    # The row is actually found through that plan, not merely planned for: an
    # index the planner will use and the predicate cannot match is no better.
    found = db_session.execute(
        text(
            "SELECT count(*) FROM passages "
            "WHERE client_name = 'acme' "
            "AND tsv @@ websearch_to_tsquery('english', 'hydrostatics')"
        )
    ).scalar_one()
    assert found == 1


def test_the_migration_is_reversible(engine):
    """`alembic downgrade -1` then `upgrade head` must both succeed.

    A migration whose downgrade is wrong is only discovered when someone needs
    it, which is the worst possible moment. Both directions are exercised here
    and the schema is left exactly as the session-scoped `engine` fixture
    migrated it.

    Isolation, following the pattern in
    tests/test_statement_timeout.py::test_postgres_actually_cancels_a_statement_over_a_tight_timeout:
    this takes its OWN connection from the engine and rolls it back in a
    `finally`, so it never touches the shared `connection` fixture's
    transaction. The rollback is not merely tidiness -- it is what undoes the
    DDL. PostgreSQL's DDL is transactional, including DROP/CREATE TABLE and
    DROP/CREATE INDEX, and Alembic's `context.begin_transaction()` is a no-op
    when the connection handed to it is already in a transaction, so nothing
    here is committed: by the end of the test `notes`, `passages`, their indexes
    and the `alembic_version` row are all back to the state the fixture left
    them in, byte for byte. The downgrade's ACCESS EXCLUSIVE locks are released
    with the same rollback, and no other connection is open at that point --
    tests run sequentially and per-test fixtures have been torn down.

    Deliberately does NOT request `db_session` or `connection`: a second open
    transaction holding a lock on `passages` would deadlock against the
    downgrade's table drop.
    """
    alembic_config = Config(str(ALEMBIC_INI))

    conn = engine.connect()
    transaction = conn.begin()
    try:
        alembic_config.attributes["connection"] = conn

        command.downgrade(alembic_config, "-1")

        # Both tables are gone, and the extension other migrations depend on is
        # not: the downgrade deliberately leaves `vector` alone because
        # `learnings` still uses it, and dropping it would break every
        # subsequent test in this run.
        assert conn.execute(text("SELECT to_regclass('notes')")).scalar() is None
        assert conn.execute(text("SELECT to_regclass('passages')")).scalar() is None
        assert conn.execute(
            text("SELECT to_regclass('learnings')")
        ).scalar() is not None
        assert conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        ).scalar() == "vector"

        command.upgrade(alembic_config, "head")

        assert conn.execute(text("SELECT to_regclass('notes')")).scalar() is not None
        assert conn.execute(
            text("SELECT to_regclass('passages')")
        ).scalar() is not None
        # The indexes come back too. The HNSW and GIN indexes are created with
        # raw op.execute() in both directions, so a typo in either is invisible
        # to the table assertions above.
        rebuilt = set(
            conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = 'passages'")
            ).scalars()
        )
        assert {
            "ix_passages_client_note",
            "ix_passages_tsv",
            "passages_embedding_idx",
        } <= rebuilt, f"indexes missing after re-upgrade: {rebuilt}"
    finally:
        transaction.rollback()
        conn.close()
