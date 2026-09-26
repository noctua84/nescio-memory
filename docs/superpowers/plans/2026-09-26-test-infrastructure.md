# Test Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `nescio-memory` a pytest suite that runs against a real PostgreSQL + pgvector container, exercises the real Alembic migration chain, and proves the per-client tenant boundary holds.

**Architecture:** A session-scoped `pgvector/pgvector:pg17` container is started by testcontainers and migrated with `alembic upgrade head`. Each test runs inside an outer transaction that is rolled back in teardown, with the ORM session joined to it via `join_transaction_mode="create_savepoint"` so the application's own `db.commit()` calls do not escape. The FastAPI app is driven through `TestClient` with `get_db` overridden to that session; the only faked component is the Ollama embedding call.

**Tech Stack:** pytest, testcontainers[postgres], SQLAlchemy 2.x, Alembic, FastAPI TestClient, PostgreSQL + pgvector, uv.

**Spec:** `docs/superpowers/specs/2026-09-26-test-infrastructure-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- Dependencies go in the **existing `dev` dependency group**, never a new group. `uv sync --locked` installs `dev` by default, and CI must not need a new flag.
- `testcontainers[postgres]>=4.15.0` — that release is where `testcontainers.community.postgres` exists as the non-deprecated path.
- `pytest>=8.0`.
- Import `PostgresContainer` from **`testcontainers.community.postgres`**, never `testcontainers.postgres` (a deprecation shim as of 4.15.0). Never use the separate `testcontainers-postgres` distribution.
- Container image: `pgvector/pgvector:pg17`, referenced through one module-level constant.
- Every dependency change is followed by `uv lock`, because `ci.yml` fails the build on a stale lockfile.
- Never add `ruff`, `mypy`, or any other checker. Never modify `app/models/learning.py`'s `created_at`/`updated_at` defaults. Never wire up `MAX_CONTENT_CHARS`/`MIN_CONTENT_CHARS`. These are explicit non-goals.
- No mocking of the database, the session, or the repository. The boundary under test is enforced by SQL; a mock would only assert we called our own methods.
- Commit prefixes follow this repo's convention: `[chore]` for tooling, `[test]` for tests, `[fix]` for bug fixes, `[impl]` for production code.
- `tests/` must never be added to by a task whose deliverable is production code, and production files must never be edited by a task whose deliverable is tests. Task 5 is the single exception and is scoped to one word.
- A missing Docker daemon must fail loudly. Never add a `skipif` that turns an unavailable container into a pass: a suite that silently skips its only integration tests reports success while verifying nothing. testcontainers already raises on a missing daemon — leave that behaviour alone.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Test dependencies in `dev`; `[tool.pytest.ini_options]` |
| `alembic/env.py` | Accept a caller-supplied `Connection` via `Config.attributes` |
| `tests/__init__.py` | Makes `tests` a package so `tests.fakes` imports unambiguously |
| `tests/conftest.py` | Env bootstrap, container, migration run, transaction isolation, client |
| `tests/fakes.py` | Deterministic embedding stand-ins; stdlib only |
| `tests/factories.py` | `make_api_key()`, `make_learning()` row builders |
| `tests/test_smoke.py` | Proves the harness itself: schema and extension exist, isolation works |
| `tests/test_auth.py` | Authentication behaviour |
| `tests/test_isolation.py` | The tenant boundary |
| `tests/test_ingest.py` | Ingest round-trip and path validation |
| `tests/test_search.py` | Ranking, `top_k`, `repo_filter` against real pgvector |
| `.github/workflows/ci.yml` | A `tests` job |

---

### Task 1: Test dependencies and pytest configuration

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (generated)
- Create: `tests/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an installed `pytest` and `testcontainers`; `pythonpath = ["."]` so that both `app.*` and `tests.*` import from the repo root.

- [ ] **Step 1: Add the dependencies**

Run from the repo root:

```bash
uv add --dev "pytest>=8.0" "testcontainers[postgres]>=4.15.0"
```

This edits the `dev` group in `pyproject.toml` and updates `uv.lock` in one step. Do not hand-edit either file.

- [ ] **Step 2: Verify the non-deprecated import path resolves**

Run:

```bash
uv run python -c "from testcontainers.community.postgres import PostgresContainer; print(PostgresContainer)"
```

Expected: `<class 'testcontainers.community.postgres.PostgresContainer'>` with **no** `DeprecationWarning`. If you see a deprecation warning, you imported `testcontainers.postgres` instead — fix the import, not the warning filter.

- [ ] **Step 3: Add pytest configuration**

Append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
# pythonpath puts the repo root on sys.path so `app.*` and `tests.*` both
# import regardless of the directory pytest was invoked from. alembic.ini's
# prepend_sys_path = "." relies on the same assumption.
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 4: Create the tests package marker**

Create `tests/__init__.py` as an empty file. It exists so `from tests.fakes import fake_embedding` is unambiguous rather than relying on namespace-package resolution.

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 5: Verify pytest runs and collects nothing yet**

Run:

```bash
uv run pytest
```

Expected: exit code 5, `no tests ran`. This is the empty-suite exit code that `ci.yml` currently warns about; it stops being a problem in Task 2.

- [ ] **Step 6: Verify the lockfile is not stale**

Run:

```bash
uv sync --locked
```

Expected: succeeds. If it reports the lockfile is out of date, run `uv lock` and retry. CI runs this exact command and fails the build otherwise.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock tests/__init__.py
git commit -m "chore: [chore] add pytest and testcontainers to the dev group"
```

---

### Task 2: Container, migrations, and the alembic/env.py enabler

**Files:**
- Modify: `alembic/env.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: Task 1's dependencies and `pythonpath`.
- Produces:
  - fixture `pg_container` (session scope) → a started `PostgresContainer`
  - fixture `engine` (session scope) → `sqlalchemy.Engine` bound to the container, schema already at `head`
  - module constant `tests.conftest.PGVECTOR_IMAGE: str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_smoke.py`:

```python
"""Proves the harness itself, before any application behaviour is asserted.

If these fail, nothing else in the suite can be trusted.
"""
from sqlalchemy import inspect, text


def test_migrations_created_every_table(engine):
    tables = set(inspect(engine).get_table_names())
    assert {"learnings", "api_keys", "alembic_version"} <= tables


def test_the_vector_extension_is_installed(engine):
    with engine.connect() as connection:
        installed = connection.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
    assert installed == 1


def test_the_embedding_column_has_the_configured_dimension(engine):
    # Migration 0002 moved this column to 384 dimensions. If create_all() were
    # ever substituted for real migrations this assertion would still pass,
    # which is why the HNSW check below exists too.
    with engine.connect() as connection:
        dimension = connection.execute(
            text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'learnings'::regclass AND attname = 'embedding'"
            )
        ).scalar()
    assert dimension == 384


def test_the_hnsw_index_exists(engine):
    # This index is created only by migration 8e6b572b0ae3 and rebuilt by 0002.
    # It is absent from the SQLAlchemy model, so its presence is proof that
    # real migrations ran rather than metadata.create_all().
    with engine.connect() as connection:
        indexes = connection.execute(
            text("SELECT indexname FROM pg_indexes WHERE tablename = 'learnings'")
        ).scalars().all()
    assert "learnings_embedding_idx" in indexes
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: FAIL — `fixture 'engine' not found`.

- [ ] **Step 3: Make `alembic/env.py` accept a caller-supplied connection**

The problem: `env.py` sets `sqlalchemy.url` from `settings.database_url` at module scope, so anything the caller sets on the `Config` object is overwritten. The fix is Alembic's own cookbook pattern — read a live `Connection` off `Config.attributes`.

Replace the existing `run_migrations_online` function in `alembic/env.py` with:

```python
def _run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    A caller may instead pass a live Connection via Config.attributes; the test
    harness uses this to migrate a container whose URL is only known at
    runtime, since the module-scope sqlalchemy.url above cannot express it.
    """
    supplied = config.attributes.get("connection", None)
    if supplied is not None:
        _run_migrations(supplied)
        return

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        _run_migrations(connection)
```

Do **not** touch the module-scope `config.set_main_option("sqlalchemy.url", settings.database_url)` line, and do not touch `run_migrations_offline`. The CLI path must behave exactly as before.

- [ ] **Step 4: Write the conftest with the container and migration fixtures**

Create `tests/conftest.py`:

```python
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
from testcontainers.community.postgres import PostgresContainer  # noqa: E402

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
```

- [ ] **Step 5: Run the test to verify it passes**

Run:

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: 4 passed. The first run pulls roughly 150 MB for the image, so allow time.

If `command.upgrade` raises `Can't locate revision` or similar, check that `ALEMBIC_INI` resolves — `alembic.ini` uses `script_location = %(here)s/alembic`, which is relative to the ini file and therefore CWD-independent.

- [ ] **Step 6: Verify the Alembic CLI path still works**

The `env.py` change must not alter normal use. Run:

```bash
uv run alembic heads
```

Expected: prints `474ac4ba2147 (head)` without error. This exercises the unchanged module-scope URL path using your local `DATABASE_URL`.

- [ ] **Step 7: Commit**

```bash
git add alembic/env.py tests/conftest.py tests/test_smoke.py
git commit -m "test: [test] run the suite against a migrated pgvector container"
```

---

### Task 3: Per-test transaction isolation

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_smoke.py`

**Interfaces:**
- Consumes: fixture `engine` from Task 2.
- Produces:
  - fixture `connection` (function scope) → `sqlalchemy.Connection` with an open outer transaction
  - fixture `db_session` (function scope) → `sqlalchemy.orm.Session` joined to it via `create_savepoint`

- [ ] **Step 1: Write the failing tests**

First extend the import block at the **top** of `tests/test_smoke.py` so it
reads:

```python
from sqlalchemy import inspect, select, text

from app.models.learning import Learning
```

Do not append imports at the bottom of the file. Then append the tests:

```python
def _probe_row(client_name: str) -> Learning:
    return Learning(
        client_name=client_name,
        repo_name="repo_a",
        file_path="probe.md",
        content="written by an isolation probe",
        meta={"chunk_index": 0},
        embedding=[0.0] * 384,
    )


def test_a_row_written_in_a_test_is_visible_within_that_test(db_session):
    db_session.add(_probe_row("visibility_probe"))
    db_session.flush()
    found = db_session.scalars(
        select(Learning).where(Learning.client_name == "visibility_probe")
    ).all()
    assert len(found) == 1


def test_an_application_level_commit_does_not_escape_the_transaction(
    db_session, engine
):
    """The ingest endpoint calls db.commit() itself.

    Under join_transaction_mode="create_savepoint" that releases a savepoint
    instead of committing the outer transaction. Proven here by reading through
    a second, independent connection while the test transaction is still open:
    the row must be visible to this test own session and invisible to everyone
    else.

    Deliberately self-contained rather than split across two ordered tests. It
    cannot pass vacuously: the first assertion fails if the row was never
    written, so the second is only ever reached with a real row in play.
    """
    db_session.add(_probe_row("commit_probe"))
    db_session.commit()

    assert db_session.scalars(
        select(Learning).where(Learning.client_name == "commit_probe")
    ).all(), "the row is not visible to the session that wrote it"

    with engine.connect() as observer:
        escaped = observer.execute(
            select(Learning.id).where(Learning.client_name == "commit_probe")
        ).all()
    assert escaped == [], "a committed row escaped the test transaction"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: both new tests FAIL with `fixture 'db_session' not found`. The four from Task 2 still pass.

- [ ] **Step 3: Add the isolation fixtures**

Add `from sqlalchemy.orm import Session  # noqa: E402` to the existing import
block in `tests/conftest.py`, beside the other `sqlalchemy` import — not at the
bottom of the file. Then append the fixtures:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_smoke.py
git commit -m "test: [test] isolate each test in a rolled-back transaction"
```

---

### Task 4: Fakes, factories, and the authenticated client fixture

**Files:**
- Create: `tests/fakes.py`
- Create: `tests/factories.py`
- Modify: `tests/conftest.py`
- Create: `tests/test_auth.py`

**Interfaces:**
- Consumes: fixture `db_session` from Task 3.
- Produces:
  - `tests.fakes.fake_embedding(text: str) -> list[float]`
  - `tests.fakes.unit_vector(axis: int) -> list[float]`
  - `tests.factories.make_api_key(db, client_name, plaintext=None, revoked=False) -> str` (returns the plaintext key to send as a header)
  - `tests.factories.make_learning(db, client_name, repo_name="repo_a", file_path="docs/note.md", content=..., embedding=None, chunk_index=0) -> Learning`
  - fixture `client` (function scope) → `TestClient` with `get_db` overridden and embeddings patched

- [ ] **Step 1: Write the failing tests**

Create `tests/test_auth.py`:

```python
"""Authentication is enforced on the api_router itself, so these assertions
hold for every /api/v1 route, not just the one used here.
"""
from tests.factories import make_api_key

SEARCH_PAYLOAD = {"query": "anything", "top_k": 5}


def test_request_without_an_api_key_is_rejected(client):
    response = client.post("/api/v1/search", json=SEARCH_PAYLOAD)
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing API key"


def test_unknown_api_key_is_rejected(client):
    response = client.post(
        "/api/v1/search",
        json=SEARCH_PAYLOAD,
        headers={"X-API-Key": "nm_definitely_not_a_real_key"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or revoked API key"


def test_revoked_api_key_is_rejected(client, db_session):
    key = make_api_key(db_session, "acme", revoked=True)
    response = client.post(
        "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or revoked API key"


def test_valid_api_key_reaches_the_endpoint(client, db_session):
    key = make_api_key(db_session, "acme")
    response = client.post(
        "/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key}
    )
    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_only_the_plaintext_key_authenticates_not_the_stored_hash(client, db_session):
    # Guards against a regression where the stored hash is compared directly to
    # the header, which would make the database contents themselves usable as
    # credentials.
    from app.core.security import _hash_key

    key = make_api_key(db_session, "acme")
    response = client.post(
        "/api/v1/search",
        json=SEARCH_PAYLOAD,
        headers={"X-API-Key": _hash_key(key)},
    )
    assert response.status_code == 401
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/test_auth.py -v
```

Expected: all FAIL — `ModuleNotFoundError: No module named 'tests.factories'`.

- [ ] **Step 3: Write the fakes**

Create `tests/fakes.py`:

```python
"""Deterministic stand-ins for the one thing these tests do not exercise.

Only the Ollama embedding call is faked. The database, the pgvector extension,
the vector(384) column, the <=> cosine operator and the HNSW index are all
real.
"""
import hashlib
import math
import random

from app.config import settings


def fake_embedding(text: str) -> list[float]:
    """A deterministic unit vector derived from `text`.

    The same text always yields the same vector and different text yields a
    different one, which is all the ingest tests need. Seeded from a SHA-256
    digest rather than Python's hash(), because the latter is salted per
    process and would make results differ between runs.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    vector = [rng.uniform(-1.0, 1.0) for _ in range(settings.embedding_dimension)]
    magnitude = math.sqrt(sum(component * component for component in vector))
    return [component / magnitude for component in vector]


def unit_vector(axis: int) -> list[float]:
    """A basis vector: 1.0 on `axis`, 0.0 elsewhere.

    Two basis vectors on different axes are orthogonal, so their cosine
    distance is exactly 1.0, and any vector's distance to itself is exactly
    0.0. That makes search ranking assertable against arithmetic we control
    rather than a model's opaque output.
    """
    vector = [0.0] * settings.embedding_dimension
    vector[axis] = 1.0
    return vector
```

- [ ] **Step 4: Write the factories**

Create `tests/factories.py`:

```python
"""Row builders for the integration suite.

These write through the test's own session, so the rows live inside the
transaction that the `connection` fixture rolls back afterwards. They flush
rather than commit: flushing makes the rows visible to subsequent queries on
the same session, including the auth dependency, without ending anything.
"""
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.security import _hash_key
from app.models.api_key import ApiKey
from app.models.learning import Learning
from tests.fakes import fake_embedding


def make_api_key(
    db: Session,
    client_name: str,
    plaintext: str | None = None,
    revoked: bool = False,
) -> str:
    """Insert an ApiKey row; return the plaintext key to send as X-API-Key.

    Hashing goes through the application's own _hash_key rather than a
    reimplementation, so that changing the hash in production breaks these
    tests instead of silently leaving them passing against a stale scheme.
    """
    plaintext = plaintext or f"nm_test_{client_name}"
    db.add(
        ApiKey(
            key_prefix=plaintext[:15],
            key_hash=_hash_key(plaintext),
            client_name=client_name,
            revoked_at=datetime.now(timezone.utc) if revoked else None,
        )
    )
    db.flush()
    return plaintext


def make_learning(
    db: Session,
    client_name: str,
    repo_name: str = "repo_a",
    file_path: str = "docs/note.md",
    content: str = "some remembered content",
    embedding: list[float] | None = None,
    chunk_index: int = 0,
) -> Learning:
    """Insert a Learning row directly, bypassing the ingest endpoint.

    client_name is set explicitly here precisely because this bypasses
    LearningRepository.add, which is what stamps it in production. Seeding
    another client's data is the whole point of the isolation tests.
    """
    row = Learning(
        client_name=client_name,
        repo_name=repo_name,
        file_path=file_path,
        content=content,
        meta={
            "file_name": file_path.rsplit("/", 1)[-1],
            "relative_path": file_path,
            "chunk_index": chunk_index,
        },
        embedding=fake_embedding(content) if embedding is None else embedding,
    )
    db.add(row)
    db.flush()
    return row
```

- [ ] **Step 5: Add the client fixture**

Add these to the existing import block in `tests/conftest.py`, after the
`testcontainers` import and keeping the `# noqa: E402` markers that every import
in this file carries (they are there because the `os.environ` block must run
first):

```python
from fastapi.testclient import TestClient  # noqa: E402

from app.api.v1 import ingest as ingest_module  # noqa: E402
from app.api.v1 import search as search_module  # noqa: E402
from app.core.db import get_db  # noqa: E402
from app.main import app  # noqa: E402
from tests.fakes import fake_embedding  # noqa: E402
```

Then append the fixture:

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/test_auth.py -v
```

Expected: 5 passed.

If `test_valid_api_key_reaches_the_endpoint` fails with a 500 rather than a 200, stop and read Task 5 — but it should pass, because an empty `learnings` table means the result-mapping comprehension never executes.

- [ ] **Step 7: Check that Langfuse's decorator is not doing network I/O**

The endpoints are wrapped in `@observe(...)`. Run:

```bash
uv run pytest tests/test_auth.py -v --durations=5
```

Expected: no test takes more than about a second beyond container startup, and no Langfuse connection errors or retry warnings appear. If you see either, add to the env bootstrap block at the top of `tests/conftest.py`:

```python
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
```

and re-run. Record in the commit message whether this was needed — the spec flags it as unverified.

- [ ] **Step 8: Commit**

```bash
git add tests/fakes.py tests/factories.py tests/conftest.py tests/test_auth.py
git commit -m "test: [test] cover API key authentication end to end"
```

---

### Task 5: Fix the search result mapping

**Files:**
- Modify: `app/api/v1/search.py` (the `SearchResult(...)` construction)
- Create: `tests/test_search.py`

**Interfaces:**
- Consumes: `client`, `db_session`, `tests.factories.make_learning`, `tests.factories.make_api_key`.
- Produces: a working `/api/v1/search` that can return rows.

**Why this task exists:** `search.py` maps rows with `l.metadata_`, but the model attribute is `meta` (the column is named `"metadata"` because that name is reserved on `DeclarativeBase`). `Learning` has no `metadata_` attribute, so any search matching at least one row raises `AttributeError` and returns 500. It has gone unnoticed because an empty table means the comprehension never runs. This is a pre-existing bug from `b552fb6`, not something the test harness introduced.

- [ ] **Step 1: Write the failing test**

Create `tests/test_search.py`:

```python
"""Search behaviour against real pgvector.

Ranking assertions use basis vectors so the cosine distances are exact rather
than approximate: distance to itself is 0.0, distance to an orthogonal vector
is 1.0, and similarity is reported as 1 - distance.
"""
import pytest

from app.api.v1 import search as search_module
from tests.factories import make_api_key, make_learning
from tests.fakes import unit_vector

# pytest, search_module and unit_vector are unused until Task 8 appends the
# ranking tests to this file. They are declared here so that task does not have
# to reopen the import block. Do not remove them as "unused".


def test_search_returns_a_stored_chunk(client, db_session):
    key = make_api_key(db_session, "acme")
    make_learning(
        db_session,
        "acme",
        repo_name="repo_a",
        file_path="docs/note.md",
        content="the chunk we expect to get back",
    )

    response = client.post(
        "/api/v1/search",
        json={"query": "the chunk we expect to get back", "top_k": 5},
        headers={"X-API-Key": key},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["content"] == "the chunk we expect to get back"
    assert results[0]["metadata"]["relative_path"] == "docs/note.md"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_search.py::test_search_returns_a_stored_chunk -v
```

Expected: FAIL. The response is a 500 and the captured output contains
`AttributeError: 'Learning' object has no attribute 'metadata_'`.

This is the bug reproducing. Confirm you see that exact AttributeError before changing anything — if you see something else, you are fixing the wrong thing.

- [ ] **Step 3: Fix the mapping**

In `app/api/v1/search.py`, in the list comprehension under the
`# Map ORM objects -> DTOs` comment, change the one attribute:

```python
    results = [
        SearchResult(content=l.content, metadata=l.meta, similarity=sim)
        for l, sim in rows
    ]
```

Change nothing else in the file. `SearchResult.metadata` is the response field name and stays as it is; only the source attribute was wrong.

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
uv run pytest tests/test_search.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Run the whole suite to check nothing regressed**

Run:

```bash
uv run pytest -v
```

Expected: all tests from Tasks 2-5 pass.

- [ ] **Step 6: Commit**

Two commits, because one is production code and the other is the test that caught it. Commit the fix first so it is reviewable on its own:

```bash
git add app/api/v1/search.py
git commit -m "fix: [fix] map search results from meta, not the nonexistent metadata_

The Learning model names the JSONB attribute meta, because metadata is
reserved on DeclarativeBase and the column is mapped explicitly. search.py
read l.metadata_, which does not exist, so any search matching at least one
row raised AttributeError and returned 500.

An empty learnings table hid this completely: the result-mapping
comprehension never executes, so search returned 200 with an empty list.
The first test to insert a row and search found it immediately.

Present since b552fb6."

git add tests/test_search.py
git commit -m "test: [test] assert search returns stored chunks"
```

---

### Task 6: The tenant isolation suite

**Files:**
- Create: `tests/test_isolation.py`

**Interfaces:**
- Consumes: `client`, `db_session`, `make_api_key`, `make_learning`.
- Produces: nothing other tasks depend on.

**Why this task exists:** `LearningRepository` enforces the tenant boundary in three places — `add()` stamps `client_name`, `search()` and `delete_by_file()` filter on it. Breaking any of them produces no error, just wrong data. This is the suite the whole harness was built for.

- [ ] **Step 1: Write the characterisation tests**

These assert behaviour the production code already has, so they are expected to
pass immediately. Step 2 is how you prove they are not passing vacuously.

Create `tests/test_isolation.py`:

```python
"""The tenant boundary.

Every failure mode here is silent in production: no exception, no error
response, just one client reading or destroying another client's data.
"""
from sqlalchemy import select

from app.models.learning import Learning
from tests.factories import make_api_key, make_learning

LONG_ENOUGH = (
    "This content comfortably exceeds the fifty character minimum that the "
    "ingest endpoint applies to each chunk."
)


def test_search_does_not_return_another_clients_learnings(client, db_session):
    make_learning(db_session, "acme", content="a private acme note")
    globex_key = make_api_key(db_session, "globex")

    response = client.post(
        "/api/v1/search",
        json={"query": "a private acme note", "top_k": 50},
        headers={"X-API-Key": globex_key},
    )

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_a_client_sees_only_its_own_rows_when_both_exist(client, db_session):
    make_learning(db_session, "acme", content="acme flavoured content")
    make_learning(db_session, "globex", content="globex flavoured content")
    globex_key = make_api_key(db_session, "globex")

    response = client.post(
        "/api/v1/search",
        json={"query": "flavoured content", "top_k": 50},
        headers={"X-API-Key": globex_key},
    )

    contents = [result["content"] for result in response.json()["results"]]
    assert contents == ["globex flavoured content"]


def test_reingest_by_another_client_does_not_delete_existing_rows(client, db_session):
    # delete_by_file filters on client_name. Without that filter, this ingest
    # would wipe acme's copy of the same path.
    make_learning(
        db_session,
        "acme",
        repo_name="shared",
        file_path="docs/note.md",
        content="acme content for the shared path",
    )
    globex_key = make_api_key(db_session, "globex")

    response = client.post(
        "/api/v1/ingest",
        data={
            "repo_name": "shared",
            "file_path": "docs/note.md",
            "content": LONG_ENOUGH,
        },
        headers={"X-API-Key": globex_key},
    )
    assert response.status_code == 200

    db_session.expire_all()
    acme_rows = db_session.scalars(
        select(Learning).where(Learning.client_name == "acme")
    ).all()
    assert len(acme_rows) == 1
    assert acme_rows[0].content == "acme content for the shared path"


def test_the_same_path_coexists_under_two_clients(client, db_session):
    acme_key = make_api_key(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")
    payload = {
        "repo_name": "shared",
        "file_path": "docs/note.md",
        "content": LONG_ENOUGH,
    }

    for key in (acme_key, globex_key):
        assert (
            client.post(
                "/api/v1/ingest", data=payload, headers={"X-API-Key": key}
            ).status_code
            == 200
        )

    db_session.expire_all()
    owners = db_session.scalars(
        select(Learning.client_name).where(Learning.file_path == "docs/note.md")
    ).all()
    assert sorted(set(owners)) == ["acme", "globex"]


def test_the_client_filter_is_what_excludes_the_row(client, db_session):
    """Two requests differing only in the key presented; one sees the row.

    This is the teeth of the isolation suite. Because the requests are
    otherwise identical, a passing pair cannot be explained by anything except
    the client_name filter, so no mutation of production code is needed to show
    that the filter is load-bearing.
    """
    make_learning(db_session, "acme", content="the contested row")
    acme_key = make_api_key(db_session, "acme")
    globex_key = make_api_key(db_session, "globex")
    payload = {"query": "the contested row", "top_k": 50}

    owner_view = client.post(
        "/api/v1/search", json=payload, headers={"X-API-Key": acme_key}
    )
    other_view = client.post(
        "/api/v1/search", json=payload, headers={"X-API-Key": globex_key}
    )

    assert [r["content"] for r in owner_view.json()["results"]] == [
        "the contested row"
    ]
    assert other_view.json()["results"] == []


def test_ingest_stamps_the_authenticated_clients_name(client, db_session):
    # LearningRepository.add stamps client_name; the endpoint never passes it.
    key = make_api_key(db_session, "acme")

    client.post(
        "/api/v1/ingest",
        data={
            "repo_name": "repo_a",
            "file_path": "docs/note.md",
            "content": LONG_ENOUGH,
        },
        headers={"X-API-Key": key},
    )

    db_session.expire_all()
    owners = db_session.scalars(select(Learning.client_name)).all()
    assert owners
    assert set(owners) == {"acme"}
```

- [ ] **Step 2: Run the tests, then prove they have teeth**

Run:

```bash
uv run pytest tests/test_isolation.py -v
```

Expected: all pass on the first run, because the production code is already
correct — Task 5's fix was the last thing in the way.

A test that has never failed has not been shown to test anything, so the suite
carries its own proof: `test_the_client_filter_is_what_excludes_the_row` issues
two requests differing **only** in which key is presented, and asserts one sees
the row while the other does not. Two otherwise-identical requests diverging
can only be explained by the `client_name` filter doing work.

Confirm that test is present and passing. Do **not** edit
`app/repositories/learning.py` to demonstrate the point: this is a test task,
production code is out of scope per the Global Constraints, and a mutation left
unrestored would ship a cross-tenant data leak.

- [ ] **Step 3: Verify the whole suite still passes**

Run:

```bash
uv run pytest -v
```

Expected: everything passes, and `git diff app/` is empty — this task must not
have touched production code at all.

- [ ] **Step 4: Commit**

```bash
git add tests/test_isolation.py
git commit -m "test: [test] pin the per-client tenant boundary"
```

---

### Task 7: Ingest round-trip and path validation

**Files:**
- Create: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `client`, `db_session`, `make_api_key`, `make_learning`.
- Produces: nothing other tasks depend on.

**Response shape warning:** `IngestResponse` declares `serialization_alias` values, and FastAPI serialises with `by_alias=True`. The JSON keys are therefore `status`, **`file`**, and **`ingested`** — not `file_path` and `chunks_ingested`. Asserting the field names instead of the aliases produces a confusing `KeyError`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ingest.py`:

```python
"""Ingest behaviour: persistence, replacement, and input validation."""
from sqlalchemy import func, select

from app.models.learning import Learning
from tests.factories import make_api_key

LONG_ENOUGH = (
    "This content comfortably exceeds the fifty character minimum that the "
    "ingest endpoint applies to each chunk."
)


def _ingest(client, key, **overrides):
    payload = {
        "repo_name": "repo_a",
        "file_path": "docs/note.md",
        "content": LONG_ENOUGH,
    }
    payload.update(overrides)
    return client.post("/api/v1/ingest", data=payload, headers={"X-API-Key": key})


def test_ingest_persists_a_chunk(client, db_session):
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key)

    assert response.status_code == 200
    body = response.json()
    # Aliases, not field names: IngestResponse sets serialization_alias and
    # FastAPI serialises by_alias.
    assert body["status"] == "success"
    assert body["file"] == "docs/note.md"
    assert body["ingested"] == 1

    db_session.expire_all()
    rows = db_session.scalars(select(Learning)).all()
    assert len(rows) == 1
    assert rows[0].content == LONG_ENOUGH
    assert rows[0].repo_name == "repo_a"
    assert rows[0].meta["relative_path"] == "docs/note.md"
    assert rows[0].meta["file_name"] == "note.md"
    assert rows[0].meta["chunk_index"] == 0
    assert len(rows[0].embedding) == 384


def test_reingesting_the_same_file_replaces_rather_than_duplicates(client, db_session):
    key = make_api_key(db_session, "acme")

    _ingest(client, key, content=LONG_ENOUGH)
    _ingest(client, key, content=LONG_ENOUGH + " Revised.")

    db_session.expire_all()
    rows = db_session.scalars(select(Learning)).all()
    assert len(rows) == 1
    assert rows[0].content.endswith("Revised.")


def test_reingesting_does_not_disturb_another_file_in_the_same_repo(client, db_session):
    key = make_api_key(db_session, "acme")

    _ingest(client, key, file_path="docs/one.md")
    _ingest(client, key, file_path="docs/two.md")
    _ingest(client, key, file_path="docs/one.md", content=LONG_ENOUGH + " Again.")

    db_session.expire_all()
    paths = db_session.scalars(select(Learning.file_path)).all()
    assert sorted(paths) == ["docs/one.md", "docs/two.md"]


def test_content_shorter_than_the_minimum_is_skipped(client, db_session):
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, content="too short to keep")

    assert response.status_code == 200
    assert response.json()["ingested"] == 0
    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Learning)) == 0


def test_long_content_is_split_into_several_chunks(client, db_session):
    # chunk_size 1000 with overlap 200 advances 800 characters per chunk, so
    # 2000 characters yields 3 windows.
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, content="x" * 2000)

    assert response.json()["ingested"] == 3
    db_session.expire_all()
    rows = db_session.scalars(select(Learning)).all()
    assert len(rows) == 3
    # chunk_index is what a consumer uses to reassemble the file in order, so
    # assert the actual values rather than just the count.
    assert sorted(row.meta["chunk_index"] for row in rows) == [0, 1, 2]


def test_an_absolute_file_path_is_rejected(client, db_session):
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, file_path="/etc/passwd")

    assert response.status_code == 400
    assert "relative path" in response.json()["detail"]
    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Learning)) == 0


def test_a_traversing_file_path_is_rejected(client, db_session):
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, file_path="docs/../../secrets.md")

    assert response.status_code == 400
    assert "relative path" in response.json()["detail"]
    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Learning)) == 0


def test_validation_runs_before_any_deletion(client, db_session):
    # _validate_file_path is called before delete_by_file. A rejected path must
    # not have destroyed an existing row on its way out.
    key = make_api_key(db_session, "acme")
    _ingest(client, key, file_path="docs/note.md")

    _ingest(client, key, file_path="/etc/passwd")

    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Learning)) == 1
```

- [ ] **Step 2: Run the tests to verify they pass or fail meaningfully**

Run:

```bash
uv run pytest tests/test_ingest.py -v
```

Expected: all pass. If `test_long_content_is_split_into_several_chunks` fails on the count, check `settings.chunk_size` and `settings.chunk_overlap` — the arithmetic in the comment assumes the defaults of 1000 and 200. Correct the expected number to match the real configuration rather than changing the configuration.

- [ ] **Step 3: Run the whole suite**

Run:

```bash
uv run pytest -v
```

Expected: everything passes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ingest.py
git commit -m "test: [test] cover the ingest round-trip and path validation"
```

---

### Task 8: Search ranking, top_k, and repo_filter

**Files:**
- Modify: `tests/test_search.py`

**Interfaces:**
- Consumes: `client`, `db_session`, `make_api_key`, `make_learning`, `unit_vector`, `app.api.v1.search`.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py`. The imports it needs are already at the top
of the file from Task 5, so do not add any:

```python
@pytest.fixture
def query_on_axis(monkeypatch):
    """Pin the query embedding to a basis vector.

    The client fixture already patched get_embedding to the deterministic fake;
    this narrows it further so the query vector is exactly known. Re-patching
    works because the endpoint resolves the module global at call time.
    """

    def _pin(axis: int) -> None:
        monkeypatch.setattr(
            search_module, "get_embedding", lambda text: unit_vector(axis)
        )

    return _pin


def test_results_are_ordered_by_cosine_distance(client, db_session, query_on_axis):
    key = make_api_key(db_session, "acme")
    make_learning(
        db_session,
        "acme",
        file_path="near.md",
        content="on axis zero",
        embedding=unit_vector(0),
    )
    make_learning(
        db_session,
        "acme",
        file_path="far.md",
        content="on axis one",
        embedding=unit_vector(1),
    )
    query_on_axis(0)

    response = client.post(
        "/api/v1/search",
        json={"query": "irrelevant, the vector is pinned", "top_k": 5},
        headers={"X-API-Key": key},
    )

    results = response.json()["results"]
    assert [result["content"] for result in results] == [
        "on axis zero",
        "on axis one",
    ]
    # Exact, not approximate: identical vectors are distance 0, orthogonal
    # vectors are distance 1, and similarity is reported as 1 - distance.
    assert results[0]["similarity"] == pytest.approx(1.0)
    assert results[1]["similarity"] == pytest.approx(0.0)


def test_top_k_limits_the_number_of_results(client, db_session, query_on_axis):
    key = make_api_key(db_session, "acme")
    for index in range(4):
        make_learning(
            db_session,
            "acme",
            file_path=f"chunk_{index}.md",
            content=f"chunk number {index}",
            embedding=unit_vector(index),
        )
    query_on_axis(0)

    response = client.post(
        "/api/v1/search",
        json={"query": "pinned", "top_k": 2},
        headers={"X-API-Key": key},
    )

    assert len(response.json()["results"]) == 2


def test_repo_filter_restricts_results_to_one_repository(
    client, db_session, query_on_axis
):
    key = make_api_key(db_session, "acme")
    make_learning(
        db_session,
        "acme",
        repo_name="repo_a",
        file_path="a.md",
        content="lives in repo a",
        embedding=unit_vector(0),
    )
    make_learning(
        db_session,
        "acme",
        repo_name="repo_b",
        file_path="b.md",
        content="lives in repo b",
        embedding=unit_vector(0),
    )
    query_on_axis(0)

    response = client.post(
        "/api/v1/search",
        json={"query": "pinned", "top_k": 50, "repo_filter": "repo_a"},
        headers={"X-API-Key": key},
    )

    contents = [result["content"] for result in response.json()["results"]]
    assert contents == ["lives in repo a"]


def test_top_k_above_the_schema_maximum_is_rejected(client, db_session):
    # SearchRequest declares top_k with le=50.
    key = make_api_key(db_session, "acme")

    response = client.post(
        "/api/v1/search",
        json={"query": "anything", "top_k": 51},
        headers={"X-API-Key": key},
    )

    assert response.status_code == 422
```

- [ ] **Step 2: Run the tests**

Run:

```bash
uv run pytest tests/test_search.py -v
```

Expected: all pass.

If ordering assertions fail, print the raw similarities to check whether the HNSW index is returning approximate neighbours. With a handful of rows Postgres will sequentially scan and the distances are exact; if you ever see approximate behaviour, assert on set membership and relative ordering rather than loosening the tolerance until it passes.

- [ ] **Step 3: Run the whole suite**

Run:

```bash
uv run pytest -v
```

Expected: everything passes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_search.py
git commit -m "test: [test] assert search ranking, top_k and repo_filter"
```

---

### Task 9: Run the suite in CI

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the whole suite.
- Produces: nothing.

- [ ] **Step 1: Add the tests job**

In `.github/workflows/ci.yml`, delete the trailing comment block that begins
`# TODO: add a test step once tests exist` along with its explanation of the
empty-suite exit code, which no longer applies. Then append this job at the
same indentation as `deps`:

```yaml
  tests:
    name: tests (py3.12)
    runs-on: ubuntu-latest
    # Tests run against a real PostgreSQL + pgvector container that
    # testcontainers starts and stops itself, so there is no `services:` block
    # and no Docker setup step -- the Docker daemon is pre-installed on hosted
    # Ubuntu runners. This job must stay on ubuntu-latest: macOS and Windows
    # hosted runners cannot run Linux containers.
    #
    # Single Python version by design. The floor is where version-specific
    # breakage shows up, and the deps job above already proves 3.12, 3.13 and
    # 3.14 all resolve and import. Three legs here would mean three container
    # pulls per push for no additional signal.
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1

      - name: Install uv
        uses: astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7  # v10.2.0
        with:
          enable-cache: true

      - name: Install dependencies (fails if uv.lock is stale)
        run: uv sync --locked --python 3.12

      - name: Run tests
        run: uv run --locked --python 3.12 pytest
```

Keep the action SHA pins exactly as they appear in the `deps` job. This repo
pins actions to commit SHAs deliberately so a retagged release cannot redirect
the workflow; Dependabot updates the SHA and the version comment together.

- [ ] **Step 2: Verify the workflow parses**

Run:

```bash
uv run python -c "
import yaml, pathlib
workflow = yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text())
print('jobs:', sorted(workflow['jobs']))
assert workflow['jobs']['tests']['runs-on'] == 'ubuntu-latest'
print('tests job parses and targets ubuntu-latest')
"
```

Expected: `jobs: ['deps', 'tests']` followed by the confirmation line.

- [ ] **Step 3: Confirm the TODO is gone**

Run:

```bash
grep -n "TODO" .github/workflows/ci.yml || echo "no TODO remains"
```

Expected: `no TODO remains`.

- [ ] **Step 4: Run the full suite one final time**

Run:

```bash
uv run pytest
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: [chore] run the test suite on the 3.12 floor"
```

---

## Verification checklist

After Task 9, confirm all of the following before considering the plan done:

- [ ] `uv run pytest` passes from a clean working tree.
- [ ] `uv sync --locked` succeeds, proving `uv.lock` is not stale.
- [ ] `uv run alembic heads` still works, proving the `env.py` change did not break the CLI path.
- [ ] `git status --porcelain` is empty.
- [ ] `git diff b552fb6..HEAD -- app/` shows exactly one production change beyond the earlier session's work: the `l.meta` fix in `search.py`.
- [ ] No `[tool.ruff]`, `[tool.mypy]`, or checker configuration was added.
- [ ] No test asserts against a mocked database, session, or repository.
