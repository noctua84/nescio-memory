"""The two STORED generated columns on `learnings` (#53, #54).

`embedding_degenerate` and `content_chars` are computed by Postgres from what
is actually on disk, never by application code (see
alembic/versions/9439c6362bb3_add_stored_chunk_stats.py and
app/models/learning.py). That is the whole point of them, so every claim here
is measured against the real pgvector container:

- what Postgres puts in the columns, including the cases Python cannot
  reproduce: a vector whose float4 norm is zero although its float64 norm is
  not (#42/#53), and a character count that is not a byte count;
- that `search()` excludes a stored degenerate row by reading the flag rather
  than recomputing a self-distance per candidate (#53);
- that the context pre-flight `measure_ranges()` reads the stored length and
  does NOT touch `learnings.content` (and its TOAST storage) at all (#54).
  That is a claim about SQL, so it is asserted against the statement the
  driver is handed, captured with a `before_cursor_execute` listener;
- that the migration backfills rows that existed before it ran. Asserted on a
  throwaway database created on the same container, migrated to
  57cfbdfa0d6a, seeded, and only then upgraded -- the shared test engine is
  never downgraded.

Rows carrying a degenerate vector are seeded directly through the factory,
never through the API: `_validated()` in app/core/embeddings.py rejects a
zero-norm embedding with a 503, so a stored degenerate row can only be a
legacy row or one written straight to the database, and that is exactly the
row these tests stand in for.
"""
import math
import re
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import make_url

from app.api.v1 import search as search_module
from app.config import settings
from app.repositories.learning import LearningRepository
from tests.factories import make_api_key, make_learning
from tests.fakes import fake_embedding, unit_vector

ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"

# The revision these columns arrive in, and the one immediately before it.
STORED_CHUNK_STATS = "9439c6362bb3"
BEFORE_STORED_CHUNK_STATS = "57cfbdfa0d6a"

DIM = settings.embedding_dimension

# Subnormal in float4 (below FLOAT32_MIN_NORMAL, 1.17549435e-38) and a
# perfectly ordinary nonzero float64. pgvector flattens it to 0 on the way
# into the column, so the stored vector has zero norm even though nothing
# Python computed about it does.
SUBNORMAL = 1e-40

ALL_ZERO = [0.0] * DIM
ALL_SUBNORMAL = [SUBNORMAL] * DIM

# Small but perfectly normal in float4: a norm of 0.01 * sqrt(1024) = 0.32.
SMALL_BUT_NORMAL = [0.01] * DIM


def _vector_literal(vector: list[float]) -> str:
    """`vector` in pgvector's text input format."""
    return "[" + ",".join(repr(float(component)) for component in vector) + "]"


def _stored(db_session, row):
    """The generated columns as Postgres computed them, re-read from the row.

    An explicit refresh rather than trusting whatever the ORM happened to
    fetch on INSERT: the contract under test is what is on disk, and after an
    UPDATE the previously loaded value is stale by definition.
    """
    db_session.refresh(row)
    return row.content_chars, row.embedding_degenerate


# ---------------------------------------------------------------------------
# content_chars: characters of the stored content, not bytes
# ---------------------------------------------------------------------------

# Each case is (content, expected character count). The multi-byte ones are
# the reason this is asserted at all: char_length counts characters while
# octet_length counts UTF-8 bytes, and the two differ for every case below
# except the ASCII one -- which is checked explicitly in the test, so a
# generation expression switched to octet_length() cannot pass.
CONTENT_LENGTH_CASES = [
    pytest.param("plain ascii chunk", 17, id="ascii"),
    pytest.param("ü", 1, id="two-byte-char"),
    pytest.param("🙂", 1, id="four-byte-char"),
    pytest.param("aü🙂", 3, id="mixed-widths"),
    pytest.param("", 0, id="empty"),
]


@pytest.mark.parametrize("content,expected", CONTENT_LENGTH_CASES)
def test_content_chars_counts_characters_not_bytes(db_session, content, expected):
    row = make_learning(db_session, "acme", content=content)
    content_chars, _ = _stored(db_session, row)

    # The independently computed expectation first: a count this test states
    # itself, not one read back out of the database.
    assert content_chars == expected
    # Then Postgres' own char_length over the same stored row, which is the
    # expression the column is defined with, and octet_length beside it so
    # the non-ASCII cases pin that the column is NOT a byte count.
    char_length, octet_length = db_session.execute(
        text("SELECT char_length(content), octet_length(content) FROM learnings WHERE id = :id"),
        {"id": row.id},
    ).one()
    assert content_chars == char_length
    if content.isascii():
        assert char_length == octet_length
    else:
        assert char_length < octet_length


def test_content_chars_is_recomputed_when_content_is_updated(db_session):
    row = make_learning(db_session, "acme", content="four")
    assert _stored(db_session, row)[0] == 4

    row.content = "aü🙂 eight"
    db_session.flush()

    content_chars, _ = _stored(db_session, row)
    assert content_chars == 9


# ---------------------------------------------------------------------------
# embedding_degenerate: zero-norm as the STORED vector, not as Python saw it
# ---------------------------------------------------------------------------

DEGENERATE_VECTORS = [
    pytest.param(ALL_ZERO, id="all-zero"),
    pytest.param(ALL_SUBNORMAL, id="all-subnormal"),
]

HEALTHY_VECTORS = [
    pytest.param(unit_vector(0), id="unit-vector"),
    pytest.param(SMALL_BUT_NORMAL, id="small-but-normal"),
    pytest.param(fake_embedding("an ordinary chunk"), id="fake-embedder-output"),
]


@pytest.mark.parametrize("vector", DEGENERATE_VECTORS)
def test_embedding_degenerate_is_true_for_a_zero_norm_stored_vector(
    db_session, vector
):
    # The all-subnormal case is the one that cannot be decided in Python:
    # every component is a nonzero float64, so its float64 norm is ordinary,
    # and it is only pgvector's float4 column that flattens it to zero. The
    # flag has to come from the stored value for that reason.
    row = make_learning(db_session, "acme", embedding=vector)
    _, degenerate = _stored(db_session, row)
    assert degenerate is True


@pytest.mark.parametrize("vector", HEALTHY_VECTORS)
def test_embedding_degenerate_is_false_for_an_ordinary_stored_vector(
    db_session, vector
):
    # The control. Without it every assertion above would be satisfied by a
    # column that was simply always true -- and search(), which filters on
    # `embedding_degenerate IS false`, would then return nothing at all.
    row = make_learning(db_session, "acme", embedding=vector)
    _, degenerate = _stored(db_session, row)
    assert degenerate is False


def test_embedding_degenerate_is_recomputed_when_the_embedding_is_updated(
    db_session,
):
    row = make_learning(db_session, "acme", embedding=unit_vector(0))
    assert _stored(db_session, row)[1] is False

    row.embedding = ALL_SUBNORMAL
    db_session.flush()
    assert _stored(db_session, row)[1] is True

    row.embedding = unit_vector(1)
    db_session.flush()
    assert _stored(db_session, row)[1] is False


def test_the_flag_agrees_with_pgvector_self_distance_on_the_stored_row(
    db_session,
):
    """The column's definition, checked against the question it stands for.

    `embedding <=> embedding` on the stored row is pgvector's own definition
    of "this vector has zero norm" (see the #42 self-distance tests in
    tests/test_search.py). The generated column is that same expression
    evaluated once at write time, so the two must never disagree -- and this
    is what would say so if a future pgvector changed either side.
    """
    rows = {
        "degenerate": make_learning(
            db_session, "acme", file_path="a.md", embedding=ALL_SUBNORMAL
        ),
        "healthy": make_learning(
            db_session, "acme", file_path="b.md", embedding=unit_vector(0)
        ),
    }
    for name, row in rows.items():
        self_distance, flag = db_session.execute(
            text(
                "SELECT embedding <=> embedding, embedding_degenerate "
                "FROM learnings WHERE id = :id"
            ),
            {"id": row.id},
        ).one()
        assert flag is math.isnan(self_distance), name
    assert rows  # both shapes were actually exercised


# ---------------------------------------------------------------------------
# search() excludes a stored degenerate row by reading the flag
# ---------------------------------------------------------------------------


@pytest.fixture
def query_on_axis(monkeypatch):
    """Pin the query embedding to a basis vector.

    The `client` fixture already patched get_embedding to the deterministic
    fake; this narrows it further so the query vector is exactly known. Axis 0
    matters: both degenerate shapes below have a zero-or-positive dot product
    with it, and the all-subnormal one lands at a clamped distance of 0.0 --
    a perfect match -- unless the stored flag excludes it.
    """

    def _pin(axis: int) -> None:
        monkeypatch.setattr(
            search_module, "get_embedding", lambda text: unit_vector(axis)
        )

    return _pin


@pytest.mark.parametrize("vector", DEGENERATE_VECTORS)
def test_search_leaves_a_stored_degenerate_row_out_of_the_response(
    client, db_session, query_on_axis, vector
):
    key = make_api_key(db_session, "acme")
    make_learning(
        db_session,
        "acme",
        file_path="healthy_0.md",
        content="the healthy row",
        embedding=unit_vector(0),
    )
    make_learning(
        db_session,
        "acme",
        file_path="healthy_1.md",
        content="the other healthy row",
        embedding=unit_vector(1),
    )
    make_learning(
        db_session,
        "acme",
        file_path="degenerate.md",
        content="the degenerate row",
        embedding=vector,
    )
    query_on_axis(0)

    response = client.post(
        "/api/v1/search",
        json={"query": "pinned", "top_k": 50},
        headers={"X-API-Key": key},
    )

    assert response.status_code == 200, response.text
    results = response.json()["results"]
    # The degenerate row is absent AND the healthy rows beside it are still
    # returned: an exclusion that dropped everything would satisfy a test
    # that only looked for the absence.
    assert [result["content"] for result in results] == [
        "the healthy row",
        "the other healthy row",
    ]
    for result in results:
        similarity = result["similarity"]
        assert isinstance(similarity, float), result
        assert math.isfinite(similarity), result


@pytest.mark.parametrize("vector", DEGENERATE_VECTORS)
def test_the_repository_search_excludes_a_stored_degenerate_row(
    db_session, vector
):
    # The same contract one level down, so a change to the route cannot be
    # mistaken for a change to the filter. Asserted on the mapped row's own
    # flag as well as on its absence, which is what ties the exclusion to the
    # stored column rather than to any query-time arithmetic.
    make_learning(
        db_session, "acme", file_path="healthy.md", embedding=unit_vector(0)
    )
    degenerate = make_learning(
        db_session, "acme", file_path="degenerate.md", embedding=vector
    )
    db_session.expire_all()

    rows = LearningRepository(db_session, "acme").search(unit_vector(0), top_k=50)

    returned = {learning.id for learning, _ in rows}
    assert degenerate.id not in returned
    assert returned, "the healthy row must still come back"
    for learning, similarity in rows:
        assert learning.embedding_degenerate is False
        assert math.isfinite(similarity)


def test_a_degenerate_row_of_another_tenant_changes_nothing(db_session):
    # The tenant filter stays top-level in search()'s WHERE clause: adding the
    # degenerate flag to it must not have turned another client's rows into
    # something this client can see or be affected by.
    make_learning(
        db_session, "other", file_path="degenerate.md", embedding=ALL_SUBNORMAL
    )
    mine = make_learning(
        db_session, "acme", file_path="mine.md", embedding=unit_vector(0)
    )
    db_session.expire_all()

    rows = LearningRepository(db_session, "acme").search(unit_vector(0), top_k=50)
    assert [learning.id for learning, _ in rows] == [mine.id]


# ---------------------------------------------------------------------------
# measure_ranges() reads the stored length, never the content
# ---------------------------------------------------------------------------


@pytest.fixture
def measure_sql(connection, monkeypatch):
    """The SQL `measure_ranges()` handed the driver, and only that SQL.

    The listener alone cannot tell the statements apart -- search(), the
    pre-flight and the fetch all select from `learnings` -- so the repository
    method is wrapped to label the statements executed inside it. Wrapping
    also means a renamed or missing `measure_ranges` fails loudly here rather
    than being read as "no content was selected".
    """
    statements: list[str] = []
    inside: list[bool] = [False]

    def before_cursor_execute(
        conn, cursor, statement, parameters, context, executemany
    ):
        if inside[0]:
            statements.append(statement)

    event.listen(connection, "before_cursor_execute", before_cursor_execute)

    original = LearningRepository.measure_ranges

    def labelled(self, *args, **kwargs):
        inside[0] = True
        try:
            return original(self, *args, **kwargs)
        finally:
            inside[0] = False

    monkeypatch.setattr(LearningRepository, "measure_ranges", labelled)

    yield statements

    event.remove(connection, "before_cursor_execute", before_cursor_execute)


# `content_chars` contains `content` as a substring, so a column reference is
# matched precisely rather than by `in`: a word boundary after `content`
# cannot match `content_chars`, since `_` is a word character.
_CONTENT_COLUMN = re.compile(r"learnings\.content\b")
_CONTENT_CHARS_COLUMN = re.compile(r"learnings\.content_chars\b")


def _seed_chunks(db_session, client_name="acme", file_path="docs/note.md", count=3):
    rows = []
    for index in range(count):
        rows.append(
            make_learning(
                db_session,
                client_name,
                file_path=file_path,
                content=f"chunk {index} " + "x" * 200,
                chunk_index=index,
                embedding=unit_vector(index),
            )
        )
    db_session.flush()
    db_session.expire_all()
    return rows


@pytest.mark.parametrize("context", ["neighbors", "document"])
def test_the_context_preflight_reads_content_chars_and_not_content(
    client, db_session, query_on_axis, measure_sql, context
):
    """#54: the pre-flight must not detoast `content` to measure it.

    Asserted against the SQL text because the regression -- summing
    `char_length(Learning.content)` again -- is completely invisible in the
    response: the numbers it produces are identical, it just reads every
    chunk's text (and its TOAST storage) to get them.
    """
    key = make_api_key(db_session, "acme")
    _seed_chunks(db_session)
    query_on_axis(0)

    response = client.post(
        "/api/v1/search",
        json={
            "query": "pinned",
            "top_k": 5,
            "context": context,
            "context_chunks": 1,
        },
        headers={"X-API-Key": key},
    )

    assert response.status_code == 200, response.text
    assert response.json()["contexts"], "context expansion must actually have run"
    assert measure_sql, "the measuring query must have run"

    for statement in measure_sql:
        assert _CONTENT_CHARS_COLUMN.search(statement), statement
        assert not _CONTENT_COLUMN.search(statement), statement
        assert "char_length(" not in statement.lower(), statement
        # The second place the tenant boundary has to appear. A leak here is
        # silent: it would mis-price another tenant's rows and withhold
        # context, which looks exactly like a tight budget.
        assert "client_name" in statement, statement


def test_the_preflight_measures_the_same_characters_it_used_to_compute(
    db_session,
):
    # Reading the stored column instead of recomputing char_length must not
    # have changed the number. The expectation is computed from the seeded
    # text in Python, so it is independent of both expressions.
    chunks = _seed_chunks(db_session, count=4)
    expected = sum(len(row.content) for row in chunks)

    rows = LearningRepository(db_session, "acme").measure_ranges(
        [("repo_a", "docs/note.md", None, None)]
    )

    assert len(rows) == 1
    assert rows[0].rows == 4
    assert rows[0].total_chars == expected


def test_the_preflight_counts_characters_not_bytes(db_session):
    # The same claim as content_chars itself, reached through the aggregate:
    # a multi-byte chunk must be priced in characters, or the budget in
    # app/core/context.py is spent against a different unit than the region
    # builder's.
    make_learning(
        db_session,
        "acme",
        file_path="docs/wide.md",
        content="aü🙂",
        chunk_index=0,
    )
    db_session.flush()

    rows = LearningRepository(db_session, "acme").measure_ranges(
        [("repo_a", "docs/wide.md", None, None)]
    )

    assert [row.total_chars for row in rows] == [3]


# ---------------------------------------------------------------------------
# The migration backfills rows that were already there
# ---------------------------------------------------------------------------


@pytest.fixture
def throwaway_database(pg_container):
    """A database of its own on the shared container, dropped afterwards.

    A separate database rather than the suite's own: this test has to stand at
    the revision BEFORE 9439c6362bb3, and downgrading the shared engine would
    drop the columns every other test in the run depends on -- and 57cfbdfa0d6a
    truncates `learnings` on the way through, so a shared downgrade/upgrade
    cycle would also be destructive.
    """
    url = make_url(pg_container.get_connection_url())
    name = f"backfill_{uuid.uuid4().hex[:12]}"

    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))

    target = create_engine(url.set(database=name))
    try:
        yield target
    finally:
        target.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def _upgrade(engine, revision: str) -> None:
    config = Config(str(ALEMBIC_INI))
    with engine.begin() as conn:
        # Alembic's documented route for a URL only known at runtime: hand
        # env.py a live Connection rather than a URL string.
        config.attributes["connection"] = conn
        command.upgrade(config, revision)


def _downgrade(engine, revision: str) -> None:
    """Run downgrades back to `revision`, the same way as _upgrade.

    ONLY ever called against a throwaway database. 9439c6362bb3 is head, so a
    downgrade to 57cfbdfa0d6a runs that one revision's downgrade() and no
    other -- 57cfbdfa0d6a's own downgrade, which truncates `learnings`, is not
    reached. Pointing this at the shared engine would still be destructive,
    because the dropped columns are what every test above reads.
    """
    config = Config(str(ALEMBIC_INI))
    with engine.begin() as conn:
        config.attributes["connection"] = conn
        command.downgrade(config, revision)


def _current_revision(engine) -> str:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()


def _columns(engine) -> set[str]:
    with engine.connect() as conn:
        return {column["name"] for column in inspect(conn).get_columns("learnings")}


def _seed_row(engine, file_path: str, content: str, vector: list[float]) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO learnings "
                "(repo_name, client_name, file_path, content, metadata, embedding) "
                "VALUES (:repo, :client, :path, :content, "
                "CAST(:meta AS jsonb), CAST(:embedding AS vector))"
            ),
            {
                "repo": "repo_a",
                "client": "acme",
                "path": file_path,
                "content": content,
                "meta": '{"chunk_index": 0}',
                "embedding": _vector_literal(vector),
            },
        )


# Everything about a row that the ALTER TABLE must leave untouched. `embedding`
# is read as text rather than as a vector so the comparison is over Postgres'
# own rendering of the stored float4s on both sides of the downgrade, with no
# driver conversion in between -- a rewrite that lost precision would show up
# here. CAST(... AS text) rather than `::text`, because text() reads a colon as
# a bind-parameter marker.
_SURVIVING_COLUMNS = text(
    "SELECT file_path, content, CAST(metadata AS text) AS metadata, "
    "CAST(embedding AS text) AS embedding, created_at, updated_at "
    "FROM learnings ORDER BY file_path"
)


def _surviving_rows(engine) -> list[tuple]:
    with engine.connect() as conn:
        return [tuple(row) for row in conn.execute(_SURVIVING_COLUMNS)]


def test_the_migration_backfills_rows_that_predate_it(throwaway_database):
    """A row written before 9439c6362bb3 gets correct stored values.

    This is the operator-facing half of the contract: the columns are STORED
    and GENERATED, so they are computed by the ALTER TABLE rewrite for every
    existing row, and nothing re-ingests. Seeded with plain SQL at the earlier
    revision -- the ORM model already declares the two columns, so it could
    not describe a pre-migration table.
    """
    _upgrade(throwaway_database, BEFORE_STORED_CHUNK_STATS)

    with throwaway_database.connect() as conn:
        columns = {column["name"] for column in inspect(conn).get_columns("learnings")}
    assert "content_chars" not in columns
    assert "embedding_degenerate" not in columns

    seeded = {
        "healthy": ("aü🙂", unit_vector(0), 3, False),
        "degenerate": ("legacy degenerate row", ALL_SUBNORMAL, 21, True),
        "all-zero": ("", ALL_ZERO, 0, True),
    }
    with throwaway_database.begin() as conn:
        for file_path, (content, vector, _, _) in seeded.items():
            conn.execute(
                text(
                    "INSERT INTO learnings "
                    "(repo_name, client_name, file_path, content, metadata, embedding) "
                    "VALUES (:repo, :client, :path, :content, '{}'::jsonb, "
                    "CAST(:embedding AS vector))"
                ),
                {
                    "repo": "repo_a",
                    "client": "acme",
                    "path": file_path,
                    "content": content,
                    "embedding": _vector_literal(vector),
                },
            )

    _upgrade(throwaway_database, STORED_CHUNK_STATS)

    with throwaway_database.connect() as conn:
        stored = {
            row.file_path: (row.content_chars, row.embedding_degenerate)
            for row in conn.execute(
                text(
                    "SELECT file_path, content_chars, embedding_degenerate "
                    "FROM learnings"
                )
            )
        }

    assert stored == {
        file_path: (chars, degenerate)
        for file_path, (_, _, chars, degenerate) in seeded.items()
    }


# ---------------------------------------------------------------------------
# ... and can be taken back off again without taking the rows with it
# ---------------------------------------------------------------------------

# The rows the round trip is measured over: one healthy, one whose float4 norm
# is zero, and one empty-content row, so both values of each generated column
# are exercised on the way back up rather than only the common case.
ROUND_TRIP_ROWS = [
    ("healthy.md", "aü\U0001f642", unit_vector(0), 3, False),
    ("degenerate.md", "legacy degenerate row", ALL_SUBNORMAL, 21, True),
    ("empty.md", "", ALL_ZERO, 0, True),
]


def test_the_migration_downgrades_without_losing_the_rows(throwaway_database):
    """downgrade() drops both columns and nothing else (#53/#54).

    This is the only test that runs this revision's downgrade at all, and it
    has to be run somewhere: a downgrade is what an operator reaches for when
    a deploy goes wrong, and a `DROP COLUMN` typo -- dropping `content`
    instead of `content_chars`, say, which is one missing suffix away -- would
    destroy the table while every other test in the suite stayed green,
    because none of them ever goes back down.

    On a throwaway database for the reason given on the fixture: downgrading
    the shared engine would drop the columns the rest of this file reads.

    Asserted as a full round trip rather than just as a downgrade, because
    half of what a downgrade is for is being able to go forward again
    afterwards -- and the second upgrade also re-exercises the backfill over
    rows that were written while the columns existed.
    """
    _upgrade(throwaway_database, STORED_CHUNK_STATS)
    for file_path, content, vector, _, _ in ROUND_TRIP_ROWS:
        _seed_row(throwaway_database, file_path, content, vector)

    expected_generated = {
        file_path: (chars, degenerate)
        for file_path, _, _, chars, degenerate in ROUND_TRIP_ROWS
    }

    def _generated() -> dict:
        with throwaway_database.connect() as conn:
            return {
                row.file_path: (row.content_chars, row.embedding_degenerate)
                for row in conn.execute(
                    text(
                        "SELECT file_path, content_chars, embedding_degenerate "
                        "FROM learnings"
                    )
                )
            }

    assert _generated() == expected_generated
    # Captured while the columns exist, and compared byte for byte afterwards.
    before = _surviving_rows(throwaway_database)
    assert len(before) == len(ROUND_TRIP_ROWS)

    _downgrade(throwaway_database, BEFORE_STORED_CHUNK_STATS)

    # The revision really moved: without this, a downgrade() that did nothing
    # at all would satisfy "the rows survived" perfectly.
    assert _current_revision(throwaway_database) == BEFORE_STORED_CHUNK_STATS
    columns = _columns(throwaway_database)
    assert "content_chars" not in columns
    assert "embedding_degenerate" not in columns
    # The columns the generation expressions READ are the ones a mistyped DROP
    # would have taken, so they are named explicitly rather than left to the
    # row comparison below.
    assert "content" in columns
    assert "embedding" in columns

    # Every row, and every other value in it, unchanged -- including the
    # embeddings, which the ADD COLUMN ... STORED rewrite had already copied
    # once and the DROP COLUMN must not touch again.
    assert _surviving_rows(throwaway_database) == before

    _upgrade(throwaway_database, STORED_CHUNK_STATS)

    assert _current_revision(throwaway_database) == STORED_CHUNK_STATS
    columns = _columns(throwaway_database)
    assert {"content_chars", "embedding_degenerate"} <= columns
    # Recomputed from what is on disk, so the values have to come back
    # identical -- a downgrade/upgrade cycle that lost a flag would leave a
    # degenerate row visible to search() for as long as nobody rewrote it.
    assert _generated() == expected_generated
    assert _surviving_rows(throwaway_database) == before
