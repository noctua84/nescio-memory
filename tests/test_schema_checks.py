"""Startup validation of the embedding column.

The check is exercised two ways: directly, with a stubbed connection, for the
failure cases; and through a real application boot against the migrated
container, to prove it does not reject a correct schema.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.core.schema_checks import verify_embedding_dimension, verify_pgvector_version
from app.main import app


class _StubConnection:
    """Returns a fixed scalar, standing in for the pg_attribute query."""

    def __init__(self, scalar):
        self._scalar = scalar

    def execute(self, *args, **kwargs):
        return self

    def scalar(self):
        return self._scalar

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _StubEngine:
    def __init__(self, scalar):
        self._scalar = scalar

    def connect(self):
        return _StubConnection(self._scalar)


class _UnreachableEngine:
    """Stands in for an engine whose connect() cannot reach the database."""

    def connect(self):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))


def test_a_matching_dimension_passes(monkeypatch):
    from app.config import settings

    verify_embedding_dimension(_StubEngine(settings.embedding_dimension))


def test_a_mismatched_dimension_raises(monkeypatch):
    from app.config import settings

    wrong = settings.embedding_dimension + 384
    with pytest.raises(RuntimeError) as excinfo:
        verify_embedding_dimension(_StubEngine(wrong))
    message = str(excinfo.value)
    # The operator needs both numbers to act on this.
    assert str(settings.embedding_dimension) in message
    assert str(wrong) in message


def test_a_missing_table_names_the_remedy():
    # A database that exists but was never migrated cannot be validated. The
    # message must point at the actual fix rather than at the column.
    with pytest.raises(RuntimeError, match="alembic upgrade head"):
        verify_embedding_dimension(_StubEngine(None))


def test_an_unreachable_database_names_the_cause():
    # A connection refused (or timed out) at boot must not surface as a raw
    # SQLAlchemy/psycopg traceback -- the operator needs the database named
    # as the cause, and the original error preserved via `from exc`.
    with pytest.raises(RuntimeError, match="database") as excinfo:
        verify_embedding_dimension(_UnreachableEngine())
    assert isinstance(excinfo.value.__cause__, OperationalError)


def test_a_supported_pgvector_version_passes():
    verify_pgvector_version(_StubEngine("0.8.6"))


def test_an_old_pgvector_version_raises():
    with pytest.raises(RuntimeError, match="0.8.0"):
        verify_pgvector_version(_StubEngine("0.7.4"))


def test_a_missing_vector_extension_names_the_remedy():
    with pytest.raises(RuntimeError, match="alembic upgrade head"):
        verify_pgvector_version(_StubEngine(None))


def test_an_unparseable_pgvector_version_raises():
    with pytest.raises(RuntimeError, match="0.8.0"):
        verify_pgvector_version(_StubEngine("not-a-version"))


def test_the_application_boots_against_the_migrated_schema(engine, monkeypatch):
    # Guards the other direction: the check must not reject a correct schema.
    # TestClient as a context manager is what runs the lifespan handler -- the
    # suite's `client` fixture does not, so this cannot be folded into it.
    import app.main as main_module

    monkeypatch.setattr(main_module, "get_engine", lambda: engine)
    with TestClient(app) as booted:
        assert booted.get("/health").status_code == 200
