# Stabilization: dependency failure handling, and six smaller fixes

**Date:** 2026-09-29
**Status:** Approved, pending implementation plan
**Target:** one release containing all seven items

## Context

`nescio-memory` reached v0.3.1 with a 30-test integration suite, Alembic-managed
schema, API key authentication and per-client scoping. What remains are the
rough edges that make the service unreliable or its correctness unverifiable.

Seven items, agreed as a single release. The first carries all the design
decisions; the other six are mechanical and are specified here only to the depth
they need.

## Item 1 — Dependency failure handling

### The problem

`app/main.py` registers no exception handlers and no middleware. The only
deliberate HTTP errors in the codebase are the two 401s in `app/core/security.py`
and the 400 in `app/api/v1/ingest.py`. Everything else reaches the client as an
unhandled `500`:

| Failure | Exception raised today |
|---|---|
| Ollama unreachable | `httpx.ConnectError` |
| Ollama slow (>30s) | `httpx.ReadTimeout` |
| Ollama returns non-2xx | `httpx.HTTPStatusError` |
| Ollama returns JSON without `embedding` | `KeyError` |
| Postgres unreachable | `sqlalchemy.exc.OperationalError` |
| `local` backend without its extra | `RuntimeError` |

A caller cannot distinguish "the service is broken" from "a dependency is
briefly down, try again", which for an agent-facing memory API is the difference
between giving up and backing off.

### D1. Fail fast with 503 and `Retry-After`

The first embedding failure aborts the whole request with `503 Service
Unavailable` and a `Retry-After` header. No retries.

Embedding is called **once per chunk**, so retries multiply against chunk count:
a 20-chunk file with three attempts each is 60 upstream calls before the caller
learns anything. Failing on the first error bounds a failed request at roughly
one Ollama timeout (~30s) instead of thirty minutes.

Retrying is safe to push onto the client because ingest is idempotent — re-ingest
deletes the client's previous chunks for that path before writing.

Rejected: `502 Bad Gateway`. More precise about whose fault it is, less
actionable for a client deciding what to do, and no natural home for
`Retry-After`.

Rejected: retry with backoff. Survives brief blips at the cost of unpredictable
request duration, on an ingest path already known to be serial and slow.

### D2. Keep the existing `{"detail": "..."}` error shape

New errors look exactly like the existing 400 and 401 responses. The status code
already discriminates, and both 503 cases call for the same client behaviour.

Rejected: adding a machine-readable `code`. No client has needed to distinguish
the two 503s, and both imply the same action. It would also leave the existing
401/400 responses inconsistent unless they migrated too.

Rejected: RFC 9457 `problem+json`. Only coherent if every existing error migrates
with it, which is a larger change than this work justifies.

### D3. Handling lives in registered exception handlers

A new `app/core/errors.py` exposes `register_exception_handlers(app)`, called
once from `create_app()`.

Rejected: `try`/`except` in each endpoint — repetitive, and every future route
has to remember. Rejected: middleware — catches everything but has to sniff
exception types with no typing help.

### D4. `httpx` must not leak into the HTTP layer

`app/core/embeddings.py` is the only module that knows Ollama is reached over
HTTP. If `main.py` caught `httpx.ConnectError` directly, that detail would leak
across two layers — and the `local` backend, which raises entirely different
exceptions, would not be covered by the same handler.

`get_embedding` therefore wraps its failures in one domain exception, defined in
`app/core/errors.py`:

```python
class EmbeddingBackendError(RuntimeError):
    """The configured embedding backend could not produce a vector."""
```

SQLAlchemy needs no equivalent: it is already the abstraction over the database,
so handling `OperationalError` directly is correct rather than leaky.

### The mapping

| Condition | Status | `detail` | `Retry-After` |
|---|---|---|---|
| Ollama unreachable, timeout, or non-2xx | 503 | `Embedding backend unavailable` | yes |
| Ollama response missing `embedding` | 503 | `Embedding backend returned an unexpected response` | yes |
| Postgres unreachable (`OperationalError`) | 503 | `Database unavailable` | yes |
| `local` backend selected, extra not installed | 500 | `Embedding backend is misconfigured` | no |

The last row is deliberate. A missing `local-embeddings` extra is a deployment
mistake, not a transient condition — it will never succeed on retry. Returning
503 with `Retry-After` would tell the client something untrue, so it gets a 500
and no header.

`Retry-After` is a fixed 30 seconds, defined as a single constant.

### Atomicity

Ingest already rolls back correctly when a chunk fails mid-file, and a caller never
observes a file half-replaced. The mechanism is that `db.commit()` is never reached
once an embedding raises, so nothing the request did is ever committed. `get_db`'s
`finally: db.close()` then releases the transaction and its row locks promptly;
verified against a real database, removing that `close()` still rolls back, but
leaves an `idle in transaction` backend behind.

This is currently an accident of how the dependency is written, asserted by no
test. The behaviour stays, and gains a test that fails embedding partway through
a multi-chunk file and asserts the pre-existing rows are untouched.

### Testing

The existing harness makes this cheap. The `client` fixture already patches
`get_embedding` in each endpoint module, so a test substitutes a function that
raises. Database failure is simulated by making the overridden session raise
`OperationalError`.

- Ollama unreachable during ingest → 503 with `Retry-After`; no rows written
- Ollama unreachable during search → 503
- Ollama response missing `embedding` → 503
- Database unavailable → 503
- `local` backend misconfigured → 500, and **no** `Retry-After`
- Failure at chunk 2 of 3 leaves previously ingested rows intact

### Out of scope

No readiness probe. `/health` stays liveness-only and deliberately does not
contact PostgreSQL or Ollama, so an unhealthy dependency cannot cause a restart.
A `/ready` endpoint that does check is a reasonable future idea and is not part
of these seven items.

No configurable Ollama timeout. With fail-fast, the existing hardcoded 30s bounds
a failed request at roughly 30s rather than 30s multiplied by chunk count.

## Items 2-7

Specified only to the depth each needs. Item 2 carries one design decision,
recorded below; the rest are mechanical.

### Item 2 — Validate `EMBEDDING_DIMENSION` at startup

`EMBEDDING_DIMENSION` is documentation only. A mismatch between the configured
value and the `vector(N)` column surfaces as a database error on insert, so
switching embedding models fails at the first ingest rather than at boot.

Validate at startup that the configured dimension matches the `learnings.embedding`
column, and fail to start if it does not. Startup is when an operator is
watching, so this converts a silent runtime failure into a loud deployment one.

**Decision: validate against the live column, accepting the coupling.**

Reading the column means querying the database during startup, so the
application will no longer start when the database is unreachable. Today it
starts fine without one, because `create_engine()` does not connect. This is a
deliberate departure, taken with the following consequences understood.

Why it is worth the coupling: the failure being prevented is not loud. A
dimension mismatch is rejected by pgvector at insert time, so today the service
starts cleanly, passes its liveness probe, serves `/health` happily, and then
fails at the first ingest — potentially long after the deploy that caused it.
Checking at startup moves that from a first-request failure to a deployment
failure, which is where a configuration error belongs.

An unreachable database, by contrast, is already loud and self-correcting: the
process exits, the orchestrator restarts it, and it comes up when the database
does. That is an ordinary crash-restart loop rather than a silent fault.

Consequences this item must handle explicitly:

- **Database unreachable at boot** — the process must fail with a message naming
  the database as the cause, not an opaque traceback.
- **`learnings` table absent** — a database that exists but has never been
  migrated cannot be validated. This must fail with a message telling the
  operator to run `alembic upgrade head`, since that is the actual remedy.
- **`/health` is unaffected.** It keeps its liveness-only contract and still
  contacts nothing. The coupling is at startup only; a database that goes away
  *after* boot must not turn a liveness probe into a restart.
- The check is read-only and runs once. It must not hold a connection open
  beyond the check itself.

### Item 3 — Enforce `MAX_CONTENT_CHARS`

`MAX_CONTENT_CHARS` (500,000) and `MIN_CONTENT_CHARS` (50) are defined in
`app/api/v1/ingest.py` and used nowhere. Request size is unbounded, and the chunk
loop compares against a literal `50`.

Enforce both: reject content over `MAX_CONTENT_CHARS` with 400, and use
`MIN_CONTENT_CHARS` in place of the literal. This changes request-rejection
behaviour, so it is a `feat:`, not a cleanup.

### Item 4 — Model/migration drift test

Nothing prevents the next divergence of the kind that left `file_path` without
its index. Add a test running `alembic.autogenerate.compare_metadata` against the
migrated test container, asserting no drift.

It needs an explicit allowlist, because two divergences are known and accepted:

- `api_keys.client_name` reflects as `TEXT` against the model's `String` —
  cosmetic in PostgreSQL
- `learnings_embedding_idx`, the HNSW index, is absent from the model by design

The allowlist is the substance of this item. It must be narrow enough to catch a
real regression and documented so a future reader knows why each entry is there.

### Item 5 — Composite index

`delete_by_file()` filters on `client_name`, `repo_name` and `file_path` and runs
on every ingest. Each column is indexed separately, so PostgreSQL combines three
indexes rather than using one.

Add a composite index over `(client_name, repo_name, file_path)` **in the model
as well as a migration**, so the drift test from item 4 stays green.

The three existing single-column indexes then become redundant, and this item
removes all three. A B-tree index serves any prefix of its columns, so the
composite already covers `client_name` alone and `client_name` with `repo_name`
— which is every access pattern in the codebase, since `search()` always filters
`client_name` and only optionally adds `repo_name`, and `delete_by_file()` uses
all three. `file_path` is never filtered on its own.

That includes dropping `ix_learnings_file_path`, added only recently in #6. That
change was correct on its own terms — it closed a real drift where the model
declared an index no migration created — and this item revisits the index
strategy as a whole now that the access patterns are written down. The model
must be updated in the same change, or item 4's drift test will fail.

### Item 6 — CI `timeout-minutes`

The `tests` job sets no `timeout-minutes`, so GitHub's 360-minute default
applies. The spec for the test harness named the failure this guards: getting an
embedding patch target wrong produces tests that make real HTTP calls and hang.
Set `timeout-minutes: 15`.

### Item 7 — Docker Hub authentication in CI

Every CI run pulls `pgvector/pgvector:pg17` and testcontainers' Ryuk reaper
anonymously. GitHub's hosted runners share egress IPs and anonymous Docker Hub
pulls are rate limited, making `toomanyrequests` the most likely cause of
intermittent red CI.

Authenticate the pull, or mirror the image. Either resolves it; the choice is an
implementation detail of this item.

## Release sequencing

Each item lands as its own small pull request to `main`. release-please
accumulates them into a single release pull request, which is merged once at the
end to cut one version containing all seven.

Item 1 goes first: items 2 and 5 touch configuration and schema that its tests
exercise. The remaining items are independent of each other.

Merge order, with the item numbers used above in parentheses so the two
numberings are not confused:

| Merge order | Item | Commit type |
|---|---|---|
| 1st | (1) Dependency failure handling | `fix:` |
| 2nd | (6) CI `timeout-minutes` | `chore:` |
| 3rd | (2) Validate `EMBEDDING_DIMENSION` | `fix:` |
| 4th | (3) Enforce `MAX_CONTENT_CHARS` | `feat:` |
| 5th | (5) Composite index | `fix:` |
| 6th | (4) Drift test and allowlist | `test:` |
| 7th | (7) Docker Hub authentication | `chore:` |

Item 5 is sequenced before item 4 deliberately: the composite index changes both
the model and the schema, so letting the drift test land afterwards means it is
written against the final index set rather than being immediately edited.

## Deliberately excluded

The eighth known rough edge — Langfuse keys in `.env` being ignored, because the
SDK reads `LANGFUSE_*` from the process environment and pydantic-settings does
not export `.env` values into it — is excluded by agreement. It is accurately
documented in the README as a limitation, and the test suite already forces
tracing off so it cannot affect test runs.
