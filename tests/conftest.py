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
# setdefault, but note the placeholder wins locally too: a developer's
# DATABASE_URL normally lives in .env, and pydantic-settings gives os.environ
# priority over .env. That is the better outcome -- local and CI runs behave
# identically and the suite never touches a real database -- but it is not
# "leaving the developer's value alone".
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://placeholder:placeholder@localhost:1/placeholder",
)

# app/api/v1/ingest.py and app/api/v1/search.py are wrapped in Langfuse's
# @observe(...) decorator, which reads its configuration directly from the
# process environment (nothing in app/ or tests/ feeds it settings.langfuse_*
# explicitly). Confirmed from the installed langfuse==4.15.4 source
# (langfuse/_client/environment_variables.py and
# langfuse/_client/client.py:363-366): the client computes
# `self._tracing_enabled = tracing_enabled and
# os.environ.get("LANGFUSE_TRACING_ENABLED", "true").lower() != "false"`,
# and when that is False its OTel tracer is replaced with a NoOpTracer, so
# @observe never attempts to export a span over the network.
#
# This is assigned unconditionally, NOT via setdefault: the whole point is to
# guarantee tracing is off during tests no matter what the outer shell/CI
# environment has exported (e.g. a developer with real
# LANGFUSE_TRACING_ENABLED=true or Langfuse credentials already in their
# environment). A setdefault here would let exactly that ambient state win
# and defeat the fix.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

import pytest  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
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
    # testcontainers defaults get_connection_url() to the older psycopg
    # (v2-style) driver name; the project now runs on psycopg 3, so this must
    # be explicit or the resulting URL would pull in a driver we no longer
    # depend on.
    with PostgresContainer(PGVECTOR_IMAGE, driver="psycopg") as container:
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
    # Force exact search for the duration of this transaction, so vector-ordering
    # assertions are deterministic.
    #
    # This was removed once, on the strength of 15 consecutive green runs, and had to
    # come back. Those runs passed only because the search was then also setting
    # hnsw.ef_search=200, which widened the HNSW candidate window enough to hide the
    # problem. With ef_search back at its default -- the right call for production --
    # the full suite went red in 4 of 6 runs, test_top_k_limits_the_number_of_results
    # returning 0 or 1 rows of 2. Measured: ef=40 without this line, 4 of 6 red;
    # ef=40 with it, 6 of 6 green.
    #
    # The mechanism: HNSW is approximate and post-filtered, so earlier tests'
    # rolled-back inserts leave index entries that crowd out the candidate window and
    # a test's own live rows never surface. The ranking tests use hand-built orthogonal
    # vectors precisely so cosine distances are exact, which an approximate index
    # cannot promise, so forcing the exact path is what they already intend. Bitmap
    # scans stay enabled, so the composite index is still exercised.
    #
    # This is a test-only setting and says nothing about production, which keeps
    # ef_search at its default deliberately. tests/test_search_recall.py opts back in
    # for itself and asserts the HNSW index is genuinely used, which is where the
    # production path gets its coverage.
    conn.execute(text("SET LOCAL enable_indexscan = off"))
    yield conn
    transaction.rollback()
    conn.close()


@pytest.fixture
def db_session(connection):
    """A Session joined to the outer transaction.

    join_transaction_mode="create_savepoint" is explicit for clarity rather than
    for effect: SQLAlchemy 2.0's default, "conditional_savepoint", resolves to
    exactly this when the bound connection already has a transaction open and
    the dialect supports SAVEPOINT, which is this fixture's situation. Removing
    the argument does not change behaviour here, and no test pins it -- do not
    read its presence as evidence that a regression in it would be caught.

    What the suite does guard is the genuinely wrong setting: with
    join_transaction_mode="control_fully", 17 tests fail, including
    test_an_application_level_commit_does_not_escape_the_transaction, because
    the application's own commit() then ends the outer transaction.

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
