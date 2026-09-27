"""Shared fixtures for the integration suite.

Every test here runs against a real PostgreSQL instance with the real pgvector
extension, started in a container. There are no database mocks: the tenant
boundary this suite exists to protect is enforced by SQL WHERE clauses and a
NOT NULL constraint, so only a real database can verify it.

See docs/superpowers/specs/2026-09-26-test-infrastructure-design.md.
"""
import os
from pathlib import Path

# This block MUST run before any `app.` import. app/config.py declares
# database_url as a required setting and builds `settings` at module scope, so
# importing any app module without DATABASE_URL raises a pydantic
# ValidationError during collection.
#
# The value is deliberately unroutable. create_engine() does not connect, so
# the module-level engine in app/core/db.py is constructed but never used once
# get_db is overridden -- but if something ever does use it, we want an
# immediate connection failure rather than a silent write to a real database.
# setdefault so a developer's own DATABASE_URL is left alone.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg2://placeholder:placeholder@localhost:1/placeholder",
)

import pytest  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from testcontainers.community.postgres import PostgresContainer  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.api.v1 import ingest as ingest_module  # noqa: E402
from app.api.v1 import search as search_module  # noqa: E402
from app.core.db import get_db  # noqa: E402
from app.main import app  # noqa: E402
from tests.fakes import fake_embedding  # noqa: E402

# testcontainers.postgres is a deprecation shim as of 4.15.0; the import above
# is the live path.

PGVECTOR_IMAGE = "pgvector/pgvector:pg17"
ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"


@pytest.fixture(scope="session")
def pg_container():
    """One container for the whole run. Readiness is handled internally by
    testcontainers, which polls with psql until the server answers."""
    with PostgresContainer(PGVECTOR_IMAGE) as container:
        yield container


@pytest.fixture(scope="session")
def engine(pg_container):
    """An engine bound to the container, with the schema migrated to head.

    Real migrations, not metadata.create_all(): this project's migrations
    truncate, backfill and rebuild an HNSW index, and that HNSW index does not
    exist in the model at all.
    """
    test_engine = create_engine(pg_container.get_connection_url())

    alembic_config = Config(str(ALEMBIC_INI))
    with test_engine.begin() as connection:
        # Alembic's documented route for a URL only known at runtime: hand
        # env.py a live Connection rather than a URL string.
        alembic_config.attributes["connection"] = connection
        command.upgrade(alembic_config, "head")

    yield test_engine

    # Dispose before the container stops, so no pooled connection outlives the
    # database it points at.
    test_engine.dispose()


@pytest.fixture
def connection(engine):
    """A connection with a real outer transaction that is always rolled back.

    This is what keeps tests from leaking into each other without paying to
    recreate the database per test.
    """
    conn = engine.connect()
    transaction = conn.begin()
    yield conn
    transaction.rollback()
    conn.close()


@pytest.fixture
def db_session(connection):
    """A Session joined to the outer transaction.

    join_transaction_mode="create_savepoint" is the load-bearing argument: the
    application calls session.commit() itself, and this mode turns that into a
    savepoint release so the outer transaction survives to be rolled back.

    Do not add a SessionEvents.after_transaction_end listener to restart the
    savepoint. That is the SQLAlchemy 1.x form of this recipe; the 2.0 docs
    state it is no longer required.
    """
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()


@pytest.fixture
def client(db_session, monkeypatch):
    """A TestClient wired to the test transaction, with embeddings faked.

    The patch targets matter. ingest.py and search.py each did
    `from app.core.embeddings import get_embedding`, which binds the name into
    their own module namespace -- patching app.core.embeddings.get_embedding
    would have no effect on what the endpoints actually call, and the tests
    would quietly make real HTTP requests and hang.
    """
    monkeypatch.setattr(ingest_module, "get_embedding", fake_embedding)
    monkeypatch.setattr(search_module, "get_embedding", fake_embedding)

    # get_current_client resolves through Depends(get_db) too, so overriding
    # get_db means a factory-created ApiKey row is immediately visible to
    # authentication -- no separate seeding path, nothing committed to clean up.
    app.dependency_overrides[get_db] = lambda: db_session
    yield TestClient(app)
    # Clearing matters: a leftover override leaks this test's closed session
    # into the next one, which then fails somewhere that points at the wrong
    # test.
    app.dependency_overrides.clear()
