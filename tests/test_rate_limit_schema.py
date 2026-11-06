"""rate_limit_windows: the migration, and the one property drift checks cannot see.

Alembic's compare_metadata compares columns, indexes and constraints, not table
prefixes, so tests/test_schema_drift.py passes whether this table is UNLOGGED or
not. Isolation follows tests/test_notes_schema.py::test_the_migration_is_reversible:
each migration runs on its own connection, in one transaction that is rolled back.
"""
from pathlib import Path

from alembic.config import Config
from sqlalchemy import text

from alembic import command
from app.models import Base

ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"
BEFORE = "c4a1e7d29b30"
THIS_REVISION = "e8d3b6f05a12"
UNLOGGED = "u"


def _persistence(connection) -> str | None:
    return connection.execute(
        text(
            "SELECT relpersistence FROM pg_class "
            "WHERE oid = to_regclass('public.rate_limit_windows')"
        )
    ).scalar_one_or_none()


def test_the_migrated_table_is_unlogged(engine):
    with engine.connect() as connection:
        assert _persistence(connection) == UNLOGGED


def test_the_model_declares_the_table_unlogged():
    # So that metadata.create_all and the migration agree, which drift cannot check.
    assert Base.metadata.tables["rate_limit_windows"]._prefixes == ["UNLOGGED"]


def test_the_migration_runs_down_and_back_up(engine):
    config = Config(str(ALEMBIC_INI))
    connection = engine.connect()
    transaction = connection.begin()
    try:
        config.attributes["connection"] = connection

        command.downgrade(config, BEFORE)
        assert _persistence(connection) is None

        command.upgrade(config, THIS_REVISION)
        assert _persistence(connection) == UNLOGGED
    finally:
        transaction.rollback()
        connection.close()
