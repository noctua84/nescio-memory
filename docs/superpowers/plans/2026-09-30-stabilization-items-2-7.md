# Stabilization Items 2-7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six remaining stabilization items so `nescio-memory` fails at deploy time rather than at request time, and so model/schema drift cannot recur silently.

**Architecture:** Two items add startup validation (a FastAPI lifespan handler for the embedding dimension; a pydantic validator already landed for `OLLAMA_URL`). Two change the schema and the test that guards it — a composite index replaces three single-column ones, and a `compare_metadata` test with an explicit allowlist prevents the next divergence. Two are CI configuration.

**Tech Stack:** FastAPI lifespan, SQLAlchemy, Alembic autogenerate, pytest against the existing testcontainers harness, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-29-stabilization-design.md`. Item 1 shipped in `b827d63`; this plan covers items 2-7.

## Global Constraints

- **One item per task, one commit per task.** Each lands as its own PR to `main`; release-please accumulates them into a single release.
- Commit prefixes, per item: `chore:` for CI, `fix:` for corrections, `feat:` for the content cap, `test:` for the drift test. Every subject carries this repo's bracket tag, e.g. `fix: [fix] ...`.
- Never add `ruff`, `mypy`, or any other checker — an explicit non-goal of the spec.
- Never add a `skipif` that turns an unavailable Docker daemon into a pass.
- **Nothing may run a database query at import time.** `tests/conftest.py` imports `app.main` at module level, before the session-scoped container exists, so any connection attempt during `create_app()` fails collection before a single test runs.
- The `client` fixture overrides `get_db` with a plain function and **does not model production teardown**. Do not use it for anything about transaction or application lifecycle.
- `/health` stays liveness-only and contacts nothing.
- The suite is at **63 tests** before this plan. Each task states its own expected total; if your arithmetic differs, report the number you actually observe rather than adjusting anything to match.

## File Structure

| File | Task | Responsibility |
|---|---|---|
| `.github/workflows/ci.yml` | 1, 6 | job timeout; authenticated image pull |
| `app/main.py` | 2 | lifespan handler calling the dimension check |
| `app/core/schema_checks.py` | 2 | **new** — the check itself, importable and unit-testable |
| `tests/test_schema_checks.py` | 2 | **new** — unit tests plus one boot test |
| `app/api/v1/ingest.py` | 3 | enforce both content constants |
| `tests/test_ingest.py` | 3 | cap and minimum coverage |
| `app/models/learning.py` | 4 | composite index replaces three single-column ones |
| `alembic/versions/<new>` | 4 | the corresponding migration |
| `tests/test_schema_drift.py` | 5 | **new** — `compare_metadata` with an allowlist |

Task order differs from item numbering; each task names its item.

---

### Task 1 (item 6): Bound the CI test job

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

**Why:** the `tests` job sets no `timeout-minutes`, so GitHub's 360-minute default applies. The harness spec named the failure this guards: getting an embedding patch target wrong produces tests that make real HTTP calls and hang. A hung job would burn six hours of runner time before anyone noticed.

- [ ] **Step 1: Add the timeout**

In `.github/workflows/ci.yml`, in the `tests` job, add `timeout-minutes: 15` directly beneath `runs-on: ubuntu-latest`, above the existing comment block:

```yaml
  tests:
    name: tests (py3.12)
    runs-on: ubuntu-latest
    # The suite completes in well under a minute; 15 is generous headroom for a
    # cold image pull. Without a bound, a test that hangs on real network I/O --
    # the documented failure mode for a mis-targeted embedding patch -- would run
    # against GitHub's 360-minute default.
    timeout-minutes: 15
```

Leave the existing explanatory comments and every step unchanged.

- [ ] **Step 2: Verify the workflow still parses**

```bash
uv run python -c "
import yaml, pathlib
w = yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text())
print('jobs:', sorted(w['jobs']))
print('tests timeout-minutes:', w['jobs']['tests']['timeout-minutes'])
assert w['jobs']['tests']['timeout-minutes'] == 15
assert w['jobs']['tests']['runs-on'] == 'ubuntu-latest'
print('ok')
"
```

Expected: `jobs: ['deps', 'tests']`, `tests timeout-minutes: 15`, `ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: [chore] bound the test job at 15 minutes"
```

---

### Task 2 (item 2): Validate the embedding dimension at startup

**Files:**
- Create: `app/core/schema_checks.py`
- Modify: `app/main.py`
- Create: `tests/test_schema_checks.py`

**Interfaces:**
- Consumes: `app.config.settings`, `app.core.db.engine`.
- Produces: `app.core.schema_checks.verify_embedding_dimension(engine) -> None`, raising `RuntimeError` on mismatch or missing table.

**Why:** `EMBEDDING_DIMENSION` is documentation only. A mismatch with the `vector(N)` column surfaces as a database error on the first ingest, so switching embedding models fails long after the deploy that caused it. The spec records the decision to validate against the live column and accept that startup then requires a reachable database.

**Why a lifespan handler and not `create_app()`:** `tests/conftest.py` imports `app.main` at module level, before the container exists. A query during `create_app()` would attempt to connect using the unroutable placeholder `DATABASE_URL` and fail collection before any test ran. A lifespan handler runs only when the application actually boots.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schema_checks.py`:

```python
"""Startup validation of the embedding column.

The check is exercised two ways: directly, with a stubbed connection, for the
failure cases; and through a real application boot against the migrated
container, to prove it does not reject a correct schema.
"""
import pytest
from fastapi.testclient import TestClient

from app.core.schema_checks import verify_embedding_dimension
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


def test_the_application_boots_against_the_migrated_schema(engine, monkeypatch):
    # Guards the other direction: the check must not reject a correct schema.
    # TestClient as a context manager is what runs the lifespan handler -- the
    # suite's `client` fixture does not, so this cannot be folded into it.
    import app.main as main_module

    monkeypatch.setattr(main_module, "engine", engine)
    with TestClient(app) as booted:
        assert booted.get("/health").status_code == 200
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_schema_checks.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'app.core.schema_checks'`.

- [ ] **Step 3: Write the check**

Create `app/core/schema_checks.py`:

```python
"""Checks run once at application startup.

These exist so a configuration error fails the deploy rather than the first
request that happens to touch it.
"""
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.config import settings

# pgvector stores a column's dimension in pg_attribute.atttypmod.
_DIMENSION_QUERY = text(
    "SELECT atttypmod FROM pg_attribute "
    "WHERE attrelid = to_regclass('learnings') AND attname = 'embedding'"
)


def verify_embedding_dimension(engine: Engine) -> None:
    """Fail fast if EMBEDDING_DIMENSION disagrees with the learnings column.

    Without this, a mismatch is rejected by pgvector at insert time, so the
    service starts cleanly, passes its liveness probe, and then fails at the
    first ingest -- possibly long after the deploy that caused it.
    """
    with engine.connect() as connection:
        actual = connection.execute(_DIMENSION_QUERY).scalar()

    if actual is None:
        # to_regclass returns NULL for a missing table, so the query yields no
        # row rather than raising. The remedy is migrating, not reconfiguring.
        raise RuntimeError(
            "The 'learnings' table or its 'embedding' column does not exist. "
            "Run `alembic upgrade head` before starting the application."
        )

    if actual != settings.embedding_dimension:
        raise RuntimeError(
            f"EMBEDDING_DIMENSION is {settings.embedding_dimension} but "
            f"learnings.embedding is vector({actual}). Every insert would fail. "
            "Either correct EMBEDDING_DIMENSION or migrate the column."
        )
```

- [ ] **Step 4: Wire it into a lifespan handler**

Rewrite `app/main.py`'s imports and `create_app`, leaving `health()` and the trailing `app = create_app()` exactly as they are:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.config import settings
from app.core.db import engine
from app.core.errors import register_exception_handlers
from app.core.schema_checks import verify_embedding_dimension
from app.helper import get_app_version

__version__ = get_app_version()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup only. Deliberately not in create_app(): tests import this module
    # before any database exists, and a query there would fail collection.
    verify_embedding_dimension(engine)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="NescioAI semantic memory core.",
        version=__version__,
        lifespan=lifespan,
    )

    register_exception_handlers(app)

    app.include_router(api_router, prefix="/api/v1")
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest tests/test_schema_checks.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Confirm nothing else regressed**

```bash
uv run pytest -q
```

Expected: 67 passed.

If instead you see errors during **collection**, the check is running at import time rather than at startup — re-read step 4.

- [ ] **Step 7: Check for OpenAPI drift**

```bash
uv run python export_openapi.py && git diff --exit-code openapi.json openapi.yaml && echo "no drift"
```

Expected: `no drift`. A lifespan handler adds no routes. On Windows the export rewrites `openapi.yaml` with CRLF while `.gitattributes` declares LF, so `git status` may show it modified while `git diff` is empty; empty diff means no drift, and regenerated spec files must not be committed.

- [ ] **Step 8: Commit**

```bash
git add app/core/schema_checks.py app/main.py tests/test_schema_checks.py
git commit -m "fix: [fix] verify the embedding dimension at startup"
```

---

### Task 3 (item 3): Enforce the content size limits

**Files:**
- Modify: `app/api/v1/ingest.py`
- Modify: `tests/test_ingest.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: a 400 response for oversized content.

**Why:** `MAX_CONTENT_CHARS` (500,000) and `MIN_CONTENT_CHARS` (50) are defined in `ingest.py` and used nowhere. Request size is unbounded, and the chunk loop compares against a literal `50`. Enforcing the cap changes which requests are rejected, so this is a `feat:`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ingest.py`:

```python
def test_content_over_the_cap_is_rejected(client, db_session):
    from app.api.v1.ingest import MAX_CONTENT_CHARS

    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, content="x" * (MAX_CONTENT_CHARS + 1))

    assert response.status_code == 400
    assert "too large" in response.json()["detail"]
    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Learning)) == 0


def test_content_exactly_at_the_cap_is_accepted(client, db_session):
    # Boundary: the cap is inclusive, so exactly MAX_CONTENT_CHARS must pass.
    from app.api.v1.ingest import MAX_CONTENT_CHARS

    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, content="x" * MAX_CONTENT_CHARS)

    assert response.status_code == 200
    assert response.json()["ingested"] > 0


def test_the_cap_is_checked_before_any_deletion(client, db_session):
    # Oversized content must not destroy what is already stored.
    key = make_api_key(db_session, "acme")
    _ingest(client, key, file_path="docs/note.md")

    from app.api.v1.ingest import MAX_CONTENT_CHARS

    _ingest(
        client,
        key,
        file_path="docs/note.md",
        content="x" * (MAX_CONTENT_CHARS + 1),
    )

    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Learning)) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_ingest.py -v
```

Expected: the two rejection tests FAIL — oversized content is currently accepted and returns 200.

- [ ] **Step 3: Enforce both constants**

In `app/api/v1/ingest.py`, add a validator beside `_validate_file_path`:

```python
def _validate_content_size(content: str) -> None:
    """Reject content above the cap before anything is deleted or embedded."""
    if len(content) > MAX_CONTENT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"content is too large: {len(content)} characters exceeds the "
                f"{MAX_CONTENT_CHARS} limit"
            ),
        )
```

In `ingest_file`, call it immediately after `_validate_file_path(file_path)` and **before** `repo.delete_by_file(...)`, so a rejected request cannot destroy existing rows:

```python
    _validate_file_path(file_path)
    _validate_content_size(content)
```

Then replace the loop's literal with the constant:

```python
        if len(chunk.strip()) < MIN_CONTENT_CHARS:
            continue
```

Change nothing else.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_ingest.py -v
```

Expected: all pass.

- [ ] **Step 5: Run the whole suite and check drift**

```bash
uv run pytest -q
uv run python export_openapi.py && git diff --exit-code openapi.json openapi.yaml && echo "no drift"
```

Expected: 70 passed, then `no drift` — the 400 is raised in the body rather than declared on the route, so the schema is unchanged.

- [ ] **Step 6: Commit**

```bash
git add app/api/v1/ingest.py tests/test_ingest.py
git commit -m "feat: [impl] enforce the ingest content size limits"
```

---

### Task 4 (item 5): Replace three single-column indexes with one composite

**Files:**
- Modify: `app/models/learning.py`
- Create: `alembic/versions/<generated>_composite_learnings_index.py`

**Interfaces:**
- Consumes: the migration chain at head `b6f6570b8822`.
- Produces: the index set that Task 5's drift test asserts against.

**Why:** `delete_by_file()` filters on `client_name`, `repo_name` and `file_path` and runs on every ingest. Three separate indexes make PostgreSQL combine them rather than use one.

A B-tree serves any **prefix** of its columns, so a composite over those three already covers every access pattern in the codebase: `search()` always filters `client_name` and only optionally adds `repo_name`; `delete_by_file()` uses all three; `file_path` is never filtered alone. All three single-column indexes therefore become redundant and are dropped.

**This reverses `ix_learnings_file_path`, added in PR #6.** That change was correct on its own terms — it closed a real drift where the model declared an index no migration created. This task revisits the index strategy now that the access patterns are written down.

- [ ] **Step 1: Update the model**

In `app/models/learning.py`, add the `Index` import and a `__table_args__`, and remove `index=True` from the three columns:

```python
from sqlalchemy import DateTime, Index, Text, func
```

```python
class Learning(Base):
    """ Learning model. """
    __tablename__ = "learnings"

    # One composite index rather than three single-column ones. A B-tree serves
    # any prefix of its columns, so this covers client_name alone (every search),
    # client_name with repo_name (a filtered search), and all three
    # (delete_by_file, which runs on every ingest). file_path is never filtered
    # on its own.
    __table_args__ = (
        Index(
            "ix_learnings_client_repo_path",
            "client_name",
            "repo_name",
            "file_path",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    repo_name: Mapped[str] = mapped_column(Text, nullable=False)
    client_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
```

Leave `content`, `meta`, `embedding`, `created_at` and `updated_at` exactly as they are.

- [ ] **Step 2: Generate the migration stub**

```bash
uv run alembic revision -m "composite learnings index"
```

- [ ] **Step 3: Fill in the migration**

```python
def upgrade() -> None:
    # Created first, so no window exists where none of these columns is indexed.
    op.create_index(
        "ix_learnings_client_repo_path",
        "learnings",
        ["client_name", "repo_name", "file_path"],
    )
    # Now redundant: the composite's prefixes cover client_name alone and
    # client_name with repo_name, and file_path is never filtered on its own.
    op.drop_index("ix_learnings_client_name", table_name="learnings")
    op.drop_index("ix_learnings_repo_name", table_name="learnings")
    op.drop_index("ix_learnings_file_path", table_name="learnings")


def downgrade() -> None:
    op.create_index("ix_learnings_client_name", "learnings", ["client_name"])
    op.create_index("ix_learnings_repo_name", "learnings", ["repo_name"])
    op.create_index("ix_learnings_file_path", "learnings", ["file_path"])
    op.drop_index("ix_learnings_client_repo_path", table_name="learnings")
```

- [ ] **Step 4: Confirm no existing test asserts the dropped indexes**

```bash
grep -rn "ix_learnings_client_name\|ix_learnings_repo_name\|ix_learnings_file_path" tests/
```

Expected: no matches. `tests/test_smoke.py` asserts `learnings_embedding_idx`,
which this task does not touch. If anything does match, report it before
proceeding — a test asserting a dropped index means either the test or this
task's premise is wrong.

- [ ] **Step 5: Verify the index set on a freshly migrated database**

The suite's container is migrated from scratch, so running it proves the chain applies. Then confirm the resulting indexes directly:

```bash
uv run pytest -q
```

Expected: 70 passed.

Then, with a `DATABASE_URL` pointing at a scratch container you migrate to head, query `pg_indexes` for `tablename = 'learnings'` and paste the result. Expected exactly: `learnings_pkey`, `learnings_embedding_idx`, `ix_learnings_client_repo_path`. Paste the real output.

- [ ] **Step 6: Verify the migration round-trips**

Against that same scratch database:

```bash
uv run alembic downgrade -1
uv run alembic upgrade head
uv run alembic heads
```

Expected: both succeed and a single head is reported. Paste the output.

- [ ] **Step 7: Commit**

```bash
git add app/models/learning.py alembic/versions/*_composite_learnings_index.py
git commit -m "fix: [fix] replace three learnings indexes with one composite"
```

---

### Task 5 (item 4): Guard against model/schema drift

**Files:**
- Create: `tests/test_schema_drift.py`

**Interfaces:**
- Consumes: the `engine` fixture, `app.models.Base`.
- Produces: nothing.

**Why:** nothing prevents the next divergence of the kind that left `file_path` without its index — a defect a 30-test suite could not see, because a missing index breaks no statement.

**The allowlist is the substance of this task.** Two divergences are known and accepted, and the test must tolerate exactly those while failing on anything else:

- `api_keys.client_name` reflects as `TEXT` against the model's `String` — cosmetic in PostgreSQL
- `learnings_embedding_idx`, the HNSW index, is absent from the model by design, because SQLAlchemy cannot express it

This task runs **after** Task 4 so it is written against the final index set.

- [ ] **Step 1: Write the test**

Create `tests/test_schema_drift.py`:

```python
"""The model and the migrations must not drift apart.

Real migrations only reveal drift that breaks an exercised statement. A missing
index breaks nothing, which is exactly how learnings.file_path went unindexed
while every test passed. This closes that gap.
"""
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from app.models import Base

# Divergences that are expected and accepted. Each entry is (operation, object
# name). Anything not listed here is a regression.
ACCEPTED_DRIFT = {
    # PostgreSQL has no practical distinction between TEXT and VARCHAR without a
    # length, so this reflects differently without meaning anything.
    ("modify_type", "api_keys.client_name"),
    # The HNSW index is created by migration and cannot be expressed in the
    # model, so autogenerate always reports it as removable. Deleting it would
    # destroy vector search performance.
    ("remove_index", "learnings_embedding_idx"),
}


def _describe(diff) -> tuple[str, str]:
    """Reduce an autogenerate diff entry to (operation, object name)."""
    if isinstance(diff, list):
        # A column alteration arrives as a list of tuples.
        diff = diff[0]
    operation = diff[0]
    if operation.endswith("_index") or operation.endswith("_constraint"):
        return operation, diff[1].name
    if operation.endswith("_table"):
        return operation, diff[1].name
    if operation.endswith("_column"):
        return operation, f"{diff[2]}.{diff[3].name}"
    # modify_* entries carry the table and column in fixed positions.
    return operation, f"{diff[2]}.{diff[3]}"


def test_the_models_and_the_migrated_schema_agree(engine):
    with engine.connect() as connection:
        diffs = compare_metadata(
            MigrationContext.configure(connection), Base.metadata
        )

    unexpected = [
        described
        for described in (_describe(diff) for diff in diffs)
        if described not in ACCEPTED_DRIFT
    ]

    assert not unexpected, (
        "model and migrations have drifted: "
        f"{unexpected}. If a divergence is genuinely intended, add it to "
        "ACCEPTED_DRIFT with a comment explaining why."
    )
```

- [ ] **Step 2: Check the diff shapes before trusting the helper**

`_describe` encodes assumptions about the tuple shapes `compare_metadata`
returns, which vary by operation. Verify them rather than assuming. Against the
migrated container, print the raw diffs:

```python
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
# connect, then:
for d in compare_metadata(MigrationContext.configure(connection), Base.metadata):
    print(repr(d))
```

Paste the real output. You should see exactly two entries, matching the two
accepted divergences. If `_describe` mis-reduces either one, correct the helper —
its job is to produce a stable `(operation, name)` pair, and the exact indices
depend on Alembic's version.

- [ ] **Step 3: Run it**

```bash
uv run pytest tests/test_schema_drift.py -v
```

Expected: 1 passed. If it fails, read the reported list: it names a real divergence between the model and the migrations, and the fix is to reconcile them, **not** to widen `ACCEPTED_DRIFT`. Widening the allowlist to make a failure go away defeats the entire point of the test.

- [ ] **Step 4: Prove it has teeth**

Temporarily add `index=True` to the `content` column in `app/models/learning.py`, re-run, and confirm the test **fails** naming `ix_learnings_content`. That is exactly the class of drift that went unnoticed before.

Revert, confirm it passes, and check `git diff app/` is empty.

- [ ] **Step 5: Run the whole suite**

```bash
uv run pytest -q
```

Expected: 71 passed, and `git diff app/` empty.

- [ ] **Step 6: Commit**

```bash
git add tests/test_schema_drift.py
git commit -m "test: [test] fail when the model and migrations drift apart"
```

---

### Task 6 (item 7): Authenticate the Docker Hub pull in CI

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: a `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` repository secret pair.
- Produces: nothing.

**Why:** every run pulls `pgvector/pgvector:pg17` and testcontainers' Ryuk reaper anonymously. GitHub's hosted runners share egress IPs and anonymous Docker Hub pulls are rate limited, making `toomanyrequests` the most likely cause of *intermittent* red CI — the kind that erodes trust in the suite.

- [ ] **Step 1: Add an authenticated login step**

In the `tests` job, insert this **before** the `Install uv` step, so the credentials are in place before anything pulls:

```yaml
      # Anonymous Docker Hub pulls are rate limited per IP, and GitHub's hosted
      # runners share egress addresses, so an unauthenticated pull of
      # pgvector/pgvector and testcontainers' Ryuk reaper fails intermittently
      # with toomanyrequests. Authenticating raises the limit substantially.
      #
      # The step is skipped when the secret is absent, so forks without it still
      # run the suite -- they simply keep the anonymous rate limit.
      - name: Log in to Docker Hub
        if: ${{ secrets.DOCKERHUB_USERNAME != '' }}
        uses: docker/login-action@<COMMIT-SHA>  # <version>
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
```

**You must resolve the SHA yourself — this plan deliberately does not supply one.**
This repo pins every action to a commit SHA with the version as a trailing
comment, so a retagged release cannot redirect the workflow. A SHA written from
memory would be worse than none, because it looks authoritative and may silently
point nowhere or somewhere hostile.

Resolve it from the source, for example:

```bash
gh api repos/docker/login-action/git/ref/tags/v3 --jq '.object.sha'
```

Follow the tag to the commit it names, confirm it belongs to `docker/login-action`,
and use the full 40-character SHA with the release version as the trailing comment,
matching the format of the two existing pins in this file. State in your report
which version you pinned and how you resolved it.

- [ ] **Step 2: Verify the workflow parses and the step is ordered correctly**

```bash
uv run python -c "
import yaml, pathlib
w = yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text())
names = [s.get('name', s.get('uses', '')) for s in w['jobs']['tests']['steps']]
print('steps:', names)
assert any('login' in str(n).lower() for n in names), 'login step missing'
login = next(i for i, n in enumerate(names) if 'login' in str(n).lower())
uv = next(i for i, n in enumerate(names) if 'Install uv' in str(n))
assert login < uv, 'login must precede the pull'
print('ok')
"
```

Expected: the step list, then `ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: [chore] authenticate the Docker Hub pull in the test job"
```

**Note for the human:** this task's benefit only materialises once `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` exist as repository secrets. The step is written to skip cleanly when they are absent, so merging it early is safe, but CI keeps the anonymous rate limit until they are set.

---

## Verification checklist

After Task 6, confirm:

- [ ] `uv run pytest` passes — 71 tests.
- [ ] `uv run python export_openapi.py && git diff --exit-code openapi.json openapi.yaml` reports no drift.
- [ ] `uv run alembic heads` reports exactly one head.
- [ ] `git status --porcelain` is empty.
- [ ] `grep -rn "MAX_CONTENT_CHARS\|MIN_CONTENT_CHARS" app/api/v1/ingest.py` shows both in use, not merely defined.
- [ ] `pg_indexes` for `learnings` on a freshly migrated database lists exactly `learnings_pkey`, `learnings_embedding_idx`, `ix_learnings_client_repo_path`.
- [ ] No `[tool.ruff]` or `[tool.mypy]` was added.
- [ ] `/health` still declares no dependencies: `grep -n "Depends" app/main.py` returns nothing.
