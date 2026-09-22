# QWEN.md — nescio-memory

Instructional context for AI agents working in this repository.

## Project Overview

**nescio-memory** is a small FastAPI service that provides *semantic memory* over a PostgreSQL +
pgvector database. It exposes two operations:

- **`POST /api/v1/ingest`** — accepts a file's text content (form-encoded), splits it into
  overlapping character chunks, embeds each chunk, and upserts them into the `learnings` table
  (deleting that file's previous chunks first, so re-ingesting is idempotent).
- **`POST /api/v1/search`** — embeds a query and returns the `top_k` nearest chunks by cosine
  similarity.

`GET /health` sits at the root, outside the version prefix, and reports the active embedding
backend and model. It is a pure liveness probe — it never contacts PostgreSQL or Ollama, so an
unhealthy dependency cannot get a pod restarted. Do not add dependency checks to it; a readiness
concern belongs in a separate endpoint.

By default embeddings are **not** computed in-process: they are requested over HTTP from an
[Ollama](https://ollama.com) `/api/embeddings` endpoint (default model `qwen3-embedding:0.6b`,
384 dims). The intended deployment is a **Kubernetes cluster that already runs an Ollama pod**, so
`OLLAMA_URL` points at an in-cluster Service — that is why HTTP is the default and why torch is not
part of the base install. An optional in-process backend sits behind `EMBEDDING_BACKEND=local` and
the `local-embeddings` extra. Langfuse `@observe` decorators are applied to both handlers for
tracing.

Stack: Python ≥ 3.12 · FastAPI · Pydantic v2 / pydantic-settings · psycopg2 + pgvector · httpx ·
Langfuse · uvicorn. Dependency management and the virtualenv are owned by **uv** (`uv.lock` is
committed and is the single source of truth).

The code lives in a single importable **`app/` package** at the repository root (no `src/`), split
by concern into `api/v1`, `core` and `schemas`. There are no migrations and no tests yet. The
package directories currently rely on **implicit namespace packages** — there are no `__init__.py`
files — which works because uvicorn is launched from the repository root, but makes `app` a rather
generic name to be sharing `sys.path` under.

> **This is a proof of concept and is expected to change significantly.** The API surface, storage
> schema, and chunking strategy are all in flux. Treat the gaps listed at the end of this document
> as known PoC rough edges to be aware of, not as a standing work queue — do not "fix" them
> unprompted, and do not assume any current shape will survive.

## Repository Layout

```
app/main.py                 create_app() factory + module-level `app`; owns GET /health
app/config.py               pydantic-settings Settings class; module-level `settings` singleton
app/helper.py               get_app_version() — importlib.metadata, else read pyproject.toml
app/api/v1/router.py        Aggregates the ingest and search routers under the /api/v1 prefix
app/api/v1/ingest.py        POST /api/v1/ingest — chunk, embed, delete-then-upsert
app/api/v1/search.py        POST /api/v1/search — embed query, cosine nearest-neighbour select
app/core/db.py              get_db_connection() — plain psycopg2 connect, one per call
app/core/embeddings.py      get_embedding() — Ollama over httpx, or in-process sentence-transformers
app/core/chunking.py        chunk_text() — sliding window; reads settings.chunk_size/overlap
app/schemas/search.py       SearchRequest, SearchResult, SearchResponse
app/schemas/ingest.py       IngestResponse
app/schemas/pagination.py   PaginationResponse — unused by design, see Known Gaps
export_openapi.py           Regenerates openapi.json + openapi.yaml from the live app
openapi.json / openapi.yaml Committed API contract; drift-checked in CI
.spectral.yaml              Spectral ruleset used by the OpenAPI lint step
README.md                   Primary user-facing docs: setup, SQL schema, API examples, limitations
LICENSE                     MIT
pyproject.toml              Metadata + runtime deps + [dependency-groups] dev
                            + [project.optional-dependencies] local-embeddings
                            No [build-system] — a virtual uv project, not a published wheel
uv.lock                     Locked dependency graph — committed on purpose, never hand-edited
.env.example                Documented configuration template
release-please-config.json  Release automation config
.release-please-manifest.json  Current released version ("0.1.0")
.github/workflows/          ci.yml, openapi.yml, release-please.yml
.gitattributes              Line-ending normalization (LF for .sh/.yml/.yaml/uv.lock)
```

## Setup and Commands

`uv` is the only supported workflow. There is no `Makefile`, no `requirements.txt`, and no
Dockerfile.

```bash
# Install dependencies into .venv (fails if uv.lock is stale vs pyproject.toml)
uv sync --locked

# Copy and fill in configuration — REQUIRED, see note below
cp .env.example .env

# Run the dev server — MUST be launched from the repository root
uv run uvicorn app.main:app --reload --port 8000

# Regenerate the OpenAPI spec after ANY route/model change (CI enforces this)
uv run python export_openapi.py

# Smoke-check the service
curl http://localhost:8000/health
```

**Critical gotcha:** `Settings.database_url` has **no default**, so importing `app.config` (and
therefore `app.main`) raises a `ValidationError` unless `DATABASE_URL` is present in `.env` or the
process environment. `export_openapi.py` needs it too — the OpenAPI workflow works around this by
injecting dummy `DATABASE_URL` / `OLLAMA_URL` env vars, because `app.openapi()` only reads route
definitions and never touches the DB. Use the same trick for any offline spec/docs generation.

**Everything is CWD-relative.** `SettingsConfigDict(env_file=".env")` resolves against the *working
directory*, not against `app/config.py` — so moving a module must not "adjust" that path to
`"../.env"`. That mistake points outside the repository, makes `.env` silently unread, and surfaces
as the `database_url` `ValidationError` above rather than as a missing-file error. The same applies
to importing `app` at all: launch uvicorn, pytest and `export_openapi.py` from the repository root.

### Required external services

| Service              | Purpose                        | Notes                                                        |
| -------------------- | ------------------------------ | ------------------------------------------------------------ |
| PostgreSQL + pgvector | storage                        | `CREATE EXTENSION IF NOT EXISTS vector;` must be run manually |
| Ollama               | embeddings                     | model + `EMBEDDING_DIMENSION` must agree (0.6b=384, 4b=1024, 8b=4096) |
| Langfuse             | tracing                        | optional; disabled when keys are empty                       |

## Database Schema

There are **no migration files** — the table must already exist. The canonical `CREATE TABLE` that
users are told to run lives in the README ("Quick start" → step 2); the shape the code actually
relies on is:

```sql
learnings (
  repo_name  text,          -- written as a column by /api/v1/ingest
  file_path  text,          -- written as a column by /api/v1/ingest
  content    text,          -- one chunk
  metadata   json/jsonb,    -- {"file_name", "relative_path", "chunk_index"}
  embedding  vector(N)      -- N == EMBEDDING_DIMENSION
)
```

`/api/v1/search` orders by `embedding <=> %s` (cosine distance) and reports `1 - distance` as
`similarity`. Rows come back from a **plain psycopg2 cursor**, i.e. as tuples in `SELECT` order
(`content`, `metadata`, `similarity`), not as dicts — `search.py` unpacks them positionally into
`SearchResult`. There is no `RealDictCursor`.

## Configuration

All settings live in `app/config.py` and are read from `.env` (`extra="ignore"`, so unknown keys are
tolerated). The `settings` object is instantiated **at import time**.

| Setting               | Default                                      | Used by                     |
| --------------------- | -------------------------------------------- | --------------------------- |
| `database_url`        | *(required)*                                 | `get_db_connection()`       |
| `embedding_backend`   | `ollama`                                     | `get_embedding()` — `Literal["ollama", "local"]`, so a typo fails at startup |
| `local_embedding_model` | `sentence-transformers/all-MiniLM-L6-v2`   | `get_embedding()`, only when the backend is `local` |
| `ollama_url`          | `http://localhost:11434/api/embeddings`       | `get_embedding()`, `ollama` backend |
| `ollama_model`        | `qwen3-embedding:0.6b`                        | `get_embedding()`, `/health` |
| `embedding_dimension` | `384`                                        | *not read in code* — must match the DB column |
| `langfuse_public_key` / `_secret_key` / `_host` | empty / `https://cloud.langfuse.com` | *not read in code* — see Known Gaps |
| `app_name`            | `Nescio Semantic Memory API`                  | `FastAPI(title=...)`        |
| `log_level`           | `INFO`                                       | *not read in code*          |
| `chunk_size` / `chunk_overlap` | `1000` / `200`                       | `chunk_text()` in `app/core/chunking.py` |
| `host` / `port`       | `localhost` / `8080`                          | *not read in code* — no `uvicorn.run()` block exists |

Note the `.env.example` / `app/config.py` mismatch: the template ships `OLLAMA_MODEL=qwen3-embedding:4b`
with `EMBEDDING_DIMENSION=1024`, while the code defaults to `0.6b` / `384`. Changing the model
**requires** recreating the `embedding` column at the matching dimension.

## CI, Releases, and Commit Conventions

Three GitHub Actions workflows:

- **`ci.yml`** — on push to `main` and all PRs. Runs `uv sync --locked` on Python **3.12, 3.13 and 3.14**
  (floor + current), then an import check of the runtime deps, then
  `uv sync --locked --extra local-embeddings --dry-run` so the opt-in extra stays resolvable without
  anyone paying for a torch download. The test step is a deliberate `TODO` because pytest exits 5 on
  an empty suite. This workflow is the guard against `uv.lock` drift: **adding or changing a
  dependency without re-running `uv lock` breaks CI.**
- **`openapi.yml`** — on push to `main` and all PRs. Syncs the **dev** dependency group, regenerates
  the spec, validates it with `openapi-spec-validator`, fails if `git diff` shows
  `openapi.json`/`openapi.yaml` changed, then lints with Spectral using `.spectral.yaml`.
  **Any edit to routes, `Form(...)` fields, or Pydantic models must be followed by
  `uv run python export_openapi.py` and committing the regenerated files.**
- **Both workflows must invoke Python tools through `uv run`.** `uv sync` populates `.venv` but does
  not activate it (setup-uv's `activate-environment` defaults to false), so a bare `python` or
  console script resolves to the runner's own interpreter and fails on import.
- **`release-please.yml`** — on push to `main`. Maintains a release PR from conventional commits;
  merging it tags the release. `bump-minor-pre-major: true` (0.x releases bump the minor), and three
  `extra-files` entries rewrite the version via jsonpath: **inside `uv.lock`** (toml), and at
  `$.info.version` in **`openapi.json`** (json) and **`openapi.yaml`** (yaml). Keep the two spec
  files in that list — `get_app_version()` feeds `info.version`, so a release that bumps
  `pyproject.toml` without also rewriting the specs fails the OpenAPI drift check on the release
  commit itself, which is exactly what happened for v0.2.0.
  `.release-please-manifest.json` and `pyproject.toml` versions must stay in sync. It runs under a
  fine-grained PAT (`secrets.RELEASE_PLEASE_TOKEN`) rather than `GITHUB_TOKEN` — with the bot token
  the release PR is authored by `github-actions`, whose `pull_request` runs GitHub holds in
  `action_required`, so `ci.yml` and `openapi.yml` would never execute on it and every release would
  merge unvalidated.

**Commit style** (from `git log`): Conventional Commits with a bracketed area tag after the colon —

```
feat: [impl] add semantic memory API with search and ingest endpoints
ci: [chore] verify locked dependencies on push and pull request
build: [chore] declare runtime dependencies and add lockfile
```

Types seen: `feat`, `fix`(unused so far), `chore`, `ci`, `build`. Tags seen: `[impl]`, `[chore]`.
Keep messages lowercase and imperative.

## Development Conventions

- **Dependencies:** edit `pyproject.toml`, then run `uv lock`; commit `pyproject.toml` **and**
  `uv.lock` together. Never hand-edit `uv.lock`. It is marked `linguist-generated=true` so GitHub
  collapses it in diffs, but lock changes are supply-chain relevant and should still be reviewed.
- **Dev-only tooling** goes in `[dependency-groups] dev` (currently `openapi-spec-validator`), not
  in `[project] dependencies` — it must not ship to runtime. Note that `uv sync` installs the `dev`
  group by default, so `ci.yml` picks it up even though only `openapi.yml` asks for it explicitly.
- **Heavy opt-in runtime features** go in `[project.optional-dependencies]` (currently
  `local-embeddings` → `sentence-transformers`), never in `[project] dependencies`. Import them
  lazily inside the function that needs them so the base install stays small, and fail with an error
  that names the install command rather than falling back silently. Keep the extra resolvable in CI
  with `uv sync --extra <name> --dry-run`, which costs nothing next to actually installing it.
- **Line endings:** `.gitattributes` stores LF and checks out native endings. `*.sh`, `*.yml`,
  `*.yaml`, and `uv.lock` are forced to LF because Linux tooling consumes them. Without this,
  Windows checkouts and the Ubuntu CI runner disagree and whole files appear modified. Do not
  delete or weaken these rules.
- **Secrets:** `.env` is gitignored; `.env.example` is the documented template. Never commit real
  keys.
- **Code style:** type hints on public helpers and Pydantic models, `X | None` union syntax,
  parameterized SQL via `%s` placeholders (no string interpolation of user input). One concern per
  module: endpoints in `app/api/v1/`, infrastructure in `app/core/`, request/response models in
  `app/schemas/`. Handlers are decorated with `@observe(name="...")` for Langfuse tracing, and
  endpoints declare `summary=`/`description=` plus `tags=[...]` so the generated spec passes the
  Spectral lint step. The old `""" ... """` section banners are gone — they belonged to the
  single-file layout.
- **Tests:** none exist yet. `ci.yml` has a placeholder showing the intended command
  (`uv run --locked pytest`). If you add the first test, also enable that CI step.
- **Docs:** `README.md` is the user-facing contract (setup, schema SQL, API examples, limitations);
  this file is the agent-facing one. When behaviour changes, update the README too — especially the
  `CREATE TABLE` block, the configuration table, and the "Current limitations" list.
- **Licensing:** MIT. New files need no license header, but any vendored third-party code must be
  MIT-compatible and attributed.

## Known Gaps and Gotchas

Verified against the current code. These are PoC-stage rough edges, **not** a work queue — treat
them as context so you don't build on a false assumption, and don't fix them unprompted. The
user-facing subset is published in the README under "Current limitations", so if you *do* fix one,
update both places.

1. **`chunk_text()`'s `or` fallbacks treat an explicit `0` as "unset".** `size = chunk_size or
   settings.chunk_size` means `chunk_text(text, chunk_size=0)` silently uses the configured size
   instead of erroring, and `overlap=0` — a legitimate "no overlap" request — silently falls back to
   `settings.chunk_overlap`. Switch to `is None` checks if that ever matters; the only current
   caller (`ingest.py`) passes neither argument.
   The non-termination hazard is **fixed**: `chunk_text()` raises `ValueError` unless
   `0 <= overlap < size`, and `Settings` enforces the same relationship in a `model_validator`, so a
   bad `.env` crash-loops the pod at startup instead of 500-ing every ingest request.
   Behavioural note: the sliding window emits a trailing short chunk (`"abcdefghij"` at size 10 /
   overlap 3 gives `['abcdefghij', 'hij']`), and it is `ingest.py` — not `chunk_text()` — that drops
   chunks under 50 characters, so `ingested` can be lower than the number of chunks produced.
2. **`app/helper.py`'s advertised `tomli` fallback does not exist.** `import tomllib` is
   unconditional at module scope, and `tomllib` is stdlib only since Python **3.11** — inside the
   project's `>=3.12` floor, so it works today on every supported and CI-tested version. But the
   comment above the read claims "built-in tomllib (Python 3.11+) or fallback to tomli" and no such
   fallback is implemented, so lowering `requires-python` below 3.11 would fail with
   `ModuleNotFoundError` at import time instead of degrading. Supporting an older floor needs
   `tomli>=2.0; python_version < "3.11"` plus a guarded import. This is the only version-sensitive
   import in the codebase — `importlib.metadata`, also used here, has been stdlib since 3.8.
3. **The `local` embeddings backend has never run against the real library.**
   `sentence-transformers` now lives in the `local-embeddings` optional extra and `get_embedding()`
   branches on `settings.embedding_backend`, but that branch has only been exercised with a stubbed
   `SentenceTransformer`. Nothing in CI installs the extra — deliberately, since torch is ~530 MB of
   Linux wheels — so a change to the real `.encode()` signature or its return dtype would go
   unnoticed until someone opts in. Verify against the real library before relying on it.
4. **Langfuse keys are not wired.** `@observe` relies on the SDK picking up `LANGFUSE_*` from the
   process environment; `settings.langfuse_*` are never passed to the SDK, and pydantic-settings
   does **not** export `.env` values into `os.environ`. Tracing from a `.env`-only setup will not
   attach.
5. **No pooling and no error handling.** Connections *are* closed properly now (`try/finally` plus
   a cursor context manager), but a fresh `psycopg2` connection is still opened per request with
   no pool, and nothing catches DB or Ollama failures — any error surfaces as a bare 500.
   `/api/v1/ingest` calls Ollama once **per chunk**, serially, inside an open transaction.
6. **`settings.host` / `port` / `log_level` / `embedding_dimension` are unused.** The server must
   be started with an explicit `uvicorn` command line, and `EMBEDDING_DIMENSION` is documentation
   only — it is never checked against the actual `vector(N)` column, so a mismatch surfaces as a
   database error at insert time rather than at startup.

**Not a gap:** `app/schemas/pagination.py` (`PaginationResponse`) has no importers and looks like
dead code, but it is deliberate scaffolding for paged retrieval — the planned `ai-os/nescio-ai`
learning loop is expected to consume bounded result sets rather than one large batch. Do **not**
remove it during cleanup or simplification passes. Its camelCase aliases (`totalCount`,
`pageSize`) differ from `IngestResponse`'s short-form aliases (`file`, `ingested`) and from
`SearchResult`, which has none; reconcile that deliberately when pagination is wired in.

### Repo hygiene

Dated `*.txt` files at the repository root (e.g.
`2026-09-18-140231-can-you-handle-the-commit-message-part-for-the-st.txt`) are exported
AI-session transcripts. They are untracked scratch artifacts — do **not** commit them, and never
use broad staging like `git add -A` here. Stage explicit paths only.
