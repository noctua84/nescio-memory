"""Startup check that the database is at the code's Alembic head.

The failure cases use stub engines (no container needed) so every shape of
"not at head" is pinned cheaply; the pass case and the lifespan wiring run
against the migrated container, because the check is only worth having if it
accepts a correctly migrated database.
"""
import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.core import schema_checks
from app.core.schema_checks import verify_migration_head
from app.main import app

# The first migration: always an ancestor of head, so "behind" stays true as
# migrations are added.
OLDEST_REVISION = "8e6b572b0ae3"


def _head() -> str:
    script = ScriptDirectory.from_config(Config(str(schema_checks._ALEMBIC_INI)))
    (head,) = script.get_heads()
    return head


class _Result:
    def __init__(self, values):
        self._values = values

    def scalar(self):
        return self._values[0]

    def scalars(self):
        return self

    def all(self):
        return list(self._values)


class _Connection:
    """Answers the two queries verify_migration_head issues, by their SQL."""

    def __init__(self, revisions):
        self._revisions = revisions  # None means no alembic_version table

    def execute(self, statement, *args, **kwargs):
        sql = str(statement)
        if "to_regclass" in sql:
            return _Result([self._revisions is not None])
        assert "FROM alembic_version" in sql, sql
        assert self._revisions is not None, "selected from a table that is missing"
        return _Result(self._revisions)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Engine:
    def __init__(self, revisions):
        self._revisions = revisions

    def connect(self):
        return _Connection(self._revisions)


class _UnreachableEngine:
    def connect(self):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))


def test_a_database_at_head_passes():
    verify_migration_head(_Engine([_head()]))


def test_a_database_behind_head_names_both_revisions_and_the_remedy():
    with pytest.raises(RuntimeError) as excinfo:
        verify_migration_head(_Engine([OLDEST_REVISION]))
    message = str(excinfo.value)
    assert OLDEST_REVISION in message
    assert _head() in message
    assert "alembic upgrade head" in message


def test_an_unknown_revision_is_rejected():
    # A database stamped by newer code than this checkout.
    with pytest.raises(RuntimeError) as excinfo:
        verify_migration_head(_Engine(["deadbeef0000"]))
    message = str(excinfo.value)
    assert "deadbeef0000" in message
    assert _head() in message
    assert "alembic upgrade head" in message


def test_a_missing_alembic_version_table_is_reported_not_raised():
    # Never migrated: a raw UndefinedTable would hide the remedy.
    with pytest.raises(RuntimeError) as excinfo:
        verify_migration_head(_Engine(None))
    assert "alembic upgrade head" in str(excinfo.value)
    assert _head() in str(excinfo.value)


def test_an_empty_alembic_version_table_is_rejected():
    with pytest.raises(RuntimeError, match="alembic upgrade head"):
        verify_migration_head(_Engine([]))


def test_an_unreachable_database_names_the_cause():
    with pytest.raises(RuntimeError, match="database") as excinfo:
        verify_migration_head(_UnreachableEngine())
    assert isinstance(excinfo.value.__cause__, OperationalError)


def test_the_migrated_container_is_at_head(engine):
    verify_migration_head(engine)


def test_the_expected_revision_does_not_depend_on_the_working_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    assert schema_checks._script_heads() == {_head()}


def test_the_lifespan_runs_the_migration_check(engine, monkeypatch):
    # Wiring: a check that exists but is never called protects nothing.
    import app.main as main_module

    def behind(_engine):
        raise RuntimeError("not at head")

    monkeypatch.setattr(main_module, "get_engine", lambda: engine)
    monkeypatch.setattr(main_module, "verify_migration_head", behind)
    with pytest.raises(RuntimeError, match="not at head"), TestClient(app):
        pass
