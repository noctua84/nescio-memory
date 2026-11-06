"""A generated column's expression must be the same in the model and the database.

tests/test_schema_drift.py compares the model with the migrated schema through
alembic's compare_metadata, and compare_metadata ignores Computed() expressions
entirely. Edit the string in app/models/passage.py, or let a migration write a
different generation expression, and that test still passes.

The column this guards, passages.embedding_degenerate, decides whether a
zero-norm vector scores as degenerate or as a perfect match (#42/#53). A silent
change to it changes search results without raising any error.

The model's expression is never compared as raw text. It is handed to Postgres
inside a temp table and read back with pg_get_expr, the same renderer that reads
the real column, so both sides are in the same form. Raw text would not match:
Postgres rewrites `::float8` as `::double precision` and qualifies operators
outside the search_path.
"""
import pytest
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

from app.models import Base

# pg_attribute joined to pg_attrdef, so a column with no default still appears
# (with a NULL expression). attgenerated is 's' for a STORED generated column.
_ATTRIBUTES = text(
    "SELECT a.attname, a.attgenerated::text AS attgenerated, "
    "pg_get_expr(d.adbin, d.adrelid) AS expr "
    "FROM pg_attribute a "
    "LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum "
    "WHERE a.attrelid = CAST(:relation AS regclass) "
    "AND a.attnum > 0 AND NOT a.attisdropped"
)

# Every Computed() column the models declare. Today that is only
# passages.embedding_degenerate; the parametrised tests pick up any future one.
COMPUTED_COLUMNS = [
    (table.name, column.name)
    for table in Base.metadata.sorted_tables
    for column in table.columns
    if column.computed is not None
]


def _attributes(conn, relation: str) -> dict[str, tuple[str, str | None]]:
    """Column name -> (attgenerated, pg_get_expr of its default or generation)."""
    return {
        row.attname: (row.attgenerated, row.expr)
        for row in conn.execute(_ATTRIBUTES, {"relation": relation}).all()
    }


def _as_postgres_renders_it(conn, table, column, sqltext: str) -> str:
    """The expression as Postgres renders it once a STORED column holds it.

    The probe copies every other column's type from the model, so `embedding`
    resolves to the same vector width the real table uses.
    """
    dialect = postgresql.dialect()
    definitions = [
        f"{other.name} {other.type.compile(dialect=dialect)}"
        for other in table.columns
        if other.computed is None
    ]
    definitions.append(
        f"{column.name} {column.type.compile(dialect=dialect)} "
        f"GENERATED ALWAYS AS ({sqltext}) STORED"
    )
    probe = f"drift_probe_{table.name}_{column.name}"
    conn.execute(text(f"CREATE TEMP TABLE {probe} ({', '.join(definitions)})"))
    try:
        return _attributes(conn, probe)[column.name][1]
    finally:
        conn.execute(text(f"DROP TABLE {probe}"))


def _drift(conn, table, column, sqltext: str) -> list[str]:
    """Every way `sqltext` for `column` disagrees with the database.

    An empty list means no drift. The database side is read from the real
    table; the model side is `sqltext`, normalised through Postgres.
    """
    problems = []

    attgenerated, database_expr = _attributes(conn, table.name)[column.name]
    if (attgenerated == "s") != (column.computed.persisted is True):
        problems.append(
            f"model persisted={column.computed.persisted!r} but the database "
            f"attgenerated is {attgenerated!r}"
        )

    model_expr = _as_postgres_renders_it(conn, table, column, sqltext)
    if model_expr != database_expr:
        problems.append(
            f"model renders {model_expr!r} but the database has {database_expr!r}"
        )
    return problems


@pytest.mark.parametrize(("table_name", "column_name"), COMPUTED_COLUMNS)
def test_the_database_generates_what_the_model_declares(
    connection, table_name, column_name
):
    table = Base.metadata.tables[table_name]
    column = table.c[column_name]

    assert _drift(connection, table, column, column.computed.sqltext.text) == []


def test_the_guard_covers_the_flag_that_exists_today():
    """An empty parametrisation would be reported as skipped, not as a pass.

    This pins the one known Computed column, so the loop above cannot quietly
    end up with nothing to check.
    """
    assert ("passages", "embedding_degenerate") in COMPUTED_COLUMNS


# Plausible edits to the flag's expression: a different constant, a negated
# predicate and a different predicate. Each is logically different.
WRONG_EXPRESSIONS = [
    "((embedding <=> embedding) = 0::float8)",
    "((embedding <=> embedding) <> 'NaN'::float8)",
    "((embedding <=> embedding) IS NULL)",
]


@pytest.mark.parametrize("wrong_expression", WRONG_EXPRESSIONS)
def test_a_different_expression_is_reported_as_drift(connection, wrong_expression):
    table = Base.metadata.tables["passages"]
    column = table.c.embedding_degenerate

    assert _drift(connection, table, column, wrong_expression)
