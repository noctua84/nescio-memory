# Test Infrastructure: pytest + testcontainers against real pgvector

**Date:** 2026-09-26
**Status:** Approved, pending implementation plan

## Context

`nescio-memory` has no tests. There is no `tests/` directory, no test dependency,
no linter, and no type checker. `.github/workflows/ci.yml` carries a standing
`# TODO: add a test step once tests exist`, with a note explaining why it was
left out: `pytest` exits 5 on an empty suite, which would have failed CI on every
run until the first test landed.

The gap became material with the six commits ending at `a8df315`, which added API
key authentication and per-client tenant scoping. `LearningRepository` now
enforces a tenant boundary in three places — `add()` stamps `client_name`, while
`search()` and `delete_by_file()` filter on it. A regression in any of those three
is silent: no exception, no error response, just one client reading or destroying
another client's data.

That boundary cannot be meaningfully tested against mocks. It is enforced by SQL
`WHERE` clauses and a `NOT NULL` constraint, so a mock of the session would assert
only that we called the methods we wrote, not that the database honours them.

## Goals

- Run tests against a real PostgreSQL instance with the real pgvector extension.
- Exercise the real Alembic migration chain, not a parallel schema definition.
- Isolate tests from each other without recreating the database per test.
- Cover the tenant boundary introduced in `4694970`, plus authentication, ingest
  round-trip, and search ranking.
- Run in CI without manual setup.

## Non-goals

These are deliberately excluded and belong in separate changes:

- `ruff` / `mypy` configuration.
- The `Learning.created_at` / `updated_at` `default=datetime.now()` bug (evaluated
  once at import, so every row receives the process start time).
- README documentation of the new authentication flow.
- Wiring up the unused `MAX_CONTENT_CHARS` / `MIN_CONTENT_CHARS` constants.

## Decisions

### D1. Schema comes from real Alembic migrations

The session fixture runs `alembic upgrade head` against the container. Tests
therefore exercise the same path a deployment does.

Rejected: `Base.metadata.create_all()`. It is faster and needs no change to
`alembic/env.py`, but it would leave the migrations with zero coverage and make
model/migration drift invisible. This project's migrations do real work — `0002`
truncates `learnings` and rebuilds the HNSW index, and `474ac4ba2147` backfills a
column before enforcing `NOT NULL`. Those are exactly the statements worth
running. The HNSW index is not expressed in the model at all, so `create_all()`
would produce a schema that lacks it.

A confirming detail: `8e6b572b0ae3` already issues
`CREATE EXTENSION IF NOT EXISTS vector`. Because we run real migrations, the
harness needs no setup SQL to enable pgvector — it arrives with the migration
chain.

Rejected: maintaining both paths as separate suites. Two schema definitions that
must be kept in agreement is the problem, not the solution.

### D2. Embeddings are faked; the database is not

`get_embedding()` posts to an Ollama HTTP endpoint. Tests replace it with a pure
function of the input text: the same text yields the same 384-dimension unit
vector, different text yields a different one.

This does not weaken the "real database" goal. The database, the pgvector
extension, the `vector(384)` column type, the `<=>` cosine operator and the HNSW
index are all genuine. Only the external model server is replaced.

For search-ranking assertions, tests insert hand-built vectors whose cosine
distances are exactly known (`[1,0,0,...]` and `[0,1,0,...]` are orthogonal,
distance exactly 1.0). This tests pgvector's real distance arithmetic against
values we fully control, which a real model's opaque output could not do.

Rejected: installing the `local-embeddings` extra to run sentence-transformers
in-process. It pulls roughly 530 MB of torch wheels and loads a model on every
run. The extra is deliberately opt-in today; this would make it mandatory to test.

Rejected: an Ollama container alongside Postgres. Closest to production and by far
the slowest, with a model pull on every cold CI run and a new source of flakiness
unrelated to our code.

### D3. Per-test isolation by transaction rollback

Each test runs inside a transaction that is rolled back in teardown. The container
and the migration run are session-scoped and paid for once.

The mechanism is SQLAlchemy 2.x's documented "joining a session into an external
transaction" recipe, using `join_transaction_mode="create_savepoint"`. That mode
matters specifically here: the ingest endpoint calls `db.commit()` itself, and
under `create_savepoint` SQLAlchemy translates that into a savepoint release
rather than committing the real outer transaction. Teardown rolls the outer
transaction back and all test data disappears, however many times the application
committed.

Note for anyone maintaining this: the `SessionEvents.after_transaction_end`
listener that restarts the savepoint, which appears in most older guides and
StackOverflow answers, is the SQLAlchemy 1.x form of this recipe. SQLAlchemy 2.0's
documentation states it is no longer required. Do not reintroduce it.

### D4. CI runs a single leg on Python 3.12

Tests run once, on the oldest interpreter permitted by
`requires-python = ">=3.12"`. The floor is where version-specific breakage
actually surfaces — newer syntax or stdlib usage that the declared floor does not
allow. The existing `deps` job already proves that 3.12, 3.13 and 3.14 all resolve
and import, so matrix coverage is not lost, and the suite pays for one container
per run instead of three.

## Architecture

### Fixture chain

```
pg_container   (session)  PostgresContainer("pgvector/pgvector:pg17")
     |
engine         (session)  create_engine(container connection url)
     |                    + alembic upgrade head via Config.attributes
     |
connection     (function) conn = engine.connect()
     |                    trans = conn.begin()          <- the real transaction
     |
db_session     (function) Session(bind=conn,
     |                            join_transaction_mode="create_savepoint")
     |                    teardown: session.close(); trans.rollback()
     |
client         (function) TestClient(app)
                          dependency_overrides[get_db] -> db_session
                          get_embedding patched in the endpoint modules
```

Session-scoped teardown runs in reverse: the engine is disposed before the
container is stopped, so no pooled connection outlives the database it points at.

Because `get_current_client` also resolves through `Depends(get_db)`, the
overridden session means an `ApiKey` row created by a test factory is immediately
visible to authentication. There is no separate seeding path and no committed
fixture data to clean up.

The `client` fixture clears `app.dependency_overrides` in teardown. Leaving an
override installed leaks the closed session of a finished test into whichever test
runs next, which fails in a way that points at the wrong test.

### Components

| File | Purpose | Depends on |
|---|---|---|
| `tests/conftest.py` | Environment bootstrap, container, migration run, session and client fixtures | testcontainers, alembic, app |
| `tests/fakes.py` | Deterministic embedding function; no network | stdlib only |
| `tests/factories.py` | `make_api_key()`, `make_learning()` row builders | app models |
| `tests/test_auth.py` | Authentication behaviour | client fixture |
| `tests/test_isolation.py` | The tenant boundary | client, factories |
| `tests/test_ingest.py` | Ingest round-trip and validation | client, factories |
| `tests/test_search.py` | Ranking and filtering against real pgvector | client, factories |

Each test module depends only on fixtures and factories, never on another test
module.

## The one production-adjacent change

`alembic/env.py` currently sets the URL unconditionally at module scope:

```python
config.set_main_option("sqlalchemy.url", settings.database_url)
```

This means test code cannot point Alembic at a container, because whatever the
caller sets on the `Config` object is overwritten at import.

The fix follows Alembic's own cookbook pattern for sharing a connection with a
programmatic command. The body that configures the context and runs the
migrations is extracted into a small `_run_migrations(connection)` helper, and
both paths call it — the existing `engine_from_config` path and the new
caller-supplied one. Nothing about how migrations execute changes; only where the
connection comes from:

```python
def run_migrations_online() -> None:
    # A caller may pass a live Connection via Config.attributes; the test harness
    # uses this to run migrations against a container whose URL is only known at
    # runtime. Falls through to the configured URL for normal CLI use.
    connectable = config.attributes.get("connection", None)
    if connectable is not None:
        _run_migrations(connectable)
        return
    # ... existing engine_from_config path, unchanged ...
```

The module-scope `set_main_option` line stays exactly as it is. The CLI path is
unaffected. An earlier sketch of this change tested whether the configured URL
still looked like the `alembic.ini` placeholder; that heuristic was rejected in
favour of the documented mechanism.

## Implementation details that are easy to get wrong

**Patch the endpoint modules, not the source module.** `ingest.py` and `search.py`
both do `from app.core.embeddings import get_embedding`, binding the name into
their own namespaces. Patching `app.core.embeddings.get_embedding` has no effect
on those bindings. The harness must patch `app.api.v1.ingest.get_embedding` and
`app.api.v1.search.get_embedding`. Getting this wrong yields tests that silently
make real HTTP calls and hang in CI.

**Set `DATABASE_URL` before importing the app.** `app/config.py` declares
`database_url: str` as required and `settings` is a module-level singleton, so
importing any `app.` module without it raises a pydantic `ValidationError` during
collection. `tests/conftest.py` sets a placeholder at module top, before any `app.` import,
using `os.environ.setdefault` so a real local `DATABASE_URL` is never clobbered.
The placeholder must be a well-formed but unroutable URL — for example
`postgresql+psycopg2://placeholder:placeholder@localhost:1/placeholder` — so that
an accidental use of the module-level engine fails immediately and visibly rather
than reaching a real database. This is safe because `create_engine()` does not
connect: the engine in `app/core/db.py` is constructed but never used once
`get_db` is overridden.

## Test inventory

### `test_auth.py`

- Request with no `X-API-Key` header returns 401.
- Request with an unknown key returns 401.
- Request with a key whose `revoked_at` is set returns 401.
- Request with a valid key reaches the endpoint.

### `test_isolation.py`

- Client B's search does not return client A's learnings.
- Client B ingesting the same `repo_name` and `file_path` as A does not delete A's
  rows.
- The same `repo_name`/`file_path` pair coexists under two different clients.

### `test_ingest.py`

- Ingest persists chunks with the authenticated client's `client_name`.
- Re-ingesting the same file replaces its chunks rather than duplicating them.
- Chunks under 50 characters after stripping are skipped.
- An absolute `file_path` returns 400.
- A `file_path` containing `..` returns 400.

### `test_search.py`

- Results are ordered by real cosine distance, asserted with orthogonal vectors.
- `top_k` limits the result count.
- `repo_filter` restricts results to one repository.

## CI

A new `tests` job in `.github/workflows/ci.yml`:

- `runs-on: ubuntu-latest`. Docker is pre-installed on GitHub's hosted Ubuntu
  runners; macOS and Windows hosted runners are not viable for testcontainers.
- Single Python version, 3.12.
- `uv sync --locked`, preserving the existing guarantee that a stale lockfile
  fails the build.
- `uv run pytest`.
- No service container and no Docker setup step: testcontainers manages the
  container lifecycle itself.
- The `# TODO: add a test step once tests exist` comment is removed, along with
  its explanation of the empty-suite exit code, which no longer applies.

## Dependencies

Added to the existing `dev` dependency group rather than a new `test` group, so
that `uv sync --locked` continues to install everything CI needs without a new
flag:

```toml
[dependency-groups]
dev = [
    "openapi-spec-validator>=0.9.0",
    "pyyaml>=6.0.3",
    "pytest>=8.0",
    "testcontainers[postgres]>=4.15.0",
]
```

The `testcontainers` floor is deliberate. `testcontainers.community.postgres` —
the non-deprecated import path — exists from 4.15.0, and Python 3.14 support
landed in 4.13.3. The separate `testcontainers-postgres` distribution on PyPI is a
vestige of the pre-4.0 package split and must not be used.

## Risks

**Langfuse's `@observe` decorator wraps `ingest_file`.** With blank keys it should
no-op, but if it instead buffers events or retries against a network endpoint it
will make the suite slow or flaky. This must be verified during implementation, and
tracing disabled explicitly in the test environment if the behaviour is not a clean
no-op. It is not safe to assume.

**The Postgres major version is unverified against production.** Nothing in this
repository pins one: there is no compose file, and `.env.example` gives only
`postgresql://user:password@localhost:5432/nescio_memory`.
`pgvector/pgvector:pg17` is a choice, not a match. If production runs a different
major, change the tag.

**Image pull cost.** The first local run and every cold CI run pulls roughly
150 MB. pgvector's documentation takes no position on tag-pinning strategy for CI,
so pinning a full version tag would be our own practice rather than project
guidance.

**Docker must be running.** A missing Docker daemon should fail loudly rather than
skip. A suite that silently skips its only integration tests is worse than one that
fails, because it reports success.

## References

- testcontainers-python 4.15.0: `testcontainers.postgres` is a deprecation shim
  re-exporting `testcontainers.community.postgres`.
  https://github.com/testcontainers/testcontainers-python
- pgvector Docker image and the standing requirement to issue
  `CREATE EXTENSION vector` per database.
  https://github.com/pgvector/pgvector
- SQLAlchemy 2.0, "Joining a Session into an External Transaction (such as for test
  suites)", including the note that event handlers to reset the nested transaction
  are no longer required.
  https://docs.sqlalchemy.org/en/20/orm/session_transaction.html
- Alembic cookbook, sharing a connection across programmatic migration commands via
  `Config.attributes`.
  https://alembic.sqlalchemy.org/en/latest/cookbook.html
- FastAPI, "Testing Dependencies with Overrides".
  https://fastapi.tiangolo.com/advanced/testing-dependencies/
- GitHub hosted runner software manifest confirming a pre-installed Docker daemon on
  Ubuntu images.
  https://github.com/actions/runner-images
