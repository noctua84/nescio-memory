# QWEN.md — nescio-memory

Instructional context for AI agents working in this repository.

## Project Overview

**nescio-memory** is a small FastAPI service that provides *semantic memory* over a PostgreSQL +
pgvector database. It stores **notes**, each split into **passages**, and exposes them under
`/api/v1` (every route requires an `X-API-Key`; all data is tenant-scoped by the key's
`client_name`):

- **`POST /api/v1/ingest`** — accepts one note as JSON (`scope`, `path_or_uri`, `content`, ...),
  splits it into passages, embeds them in batches, and writes the note and its passages in one
  transaction. Re-ingesting an unchanged note is skipped unless `force` is set.
- **`GET /api/v1/notes/{id}`** and **`DELETE /api/v1/notes/{id}`** — read or erase one note (the
  delete cascades to its passages). A missing note and another tenant's note give the same `404`.
- **`POST /api/v1/search`** — hybrid search: a full-text branch and a vector branch fused by
  reciprocal rank (RRF) in one SQL statement. Each result carries `score`, `fts_rank` and
  `vector_rank`; the response carries a top-level `degraded` flag, set when the query could not be
  embedded and the search fell back to keyword-only.

`GET /health` sits at the root, outside the version prefix, and reports the active embedding
backend and model. It is a pure liveness probe — it never contacts PostgreSQL or Ollama, so an
unhealthy dependency cannot get a pod restarted. Do not add dependency checks to it; a readiness
concern belongs in a separate endpoint.

By default embeddings are **not** computed in-process: they are requested over HTTP from an
[Ollama](https://ollama.com) `/api/embed` batch endpoint (default model `qwen3-embedding:0.6b`,
1024 dims). The intended deployment is a **Kubernetes cluster that already runs an Ollama pod**, so
`OLLAMA_URL` points at an in-cluster Service — that is why HTTP is the default and why torch is not
part of the base install. An optional in-process backend sits behind `EMBEDDING_BACKEND=local` and
the `local-embeddings` extra. Langfuse `@observe` decorators are applied to the handlers for
tracing, with input and output capture off.

Stack: Python ≥ 3.12 · FastAPI · Pydantic v2 / pydantic-settings · SQLAlchemy 2 + psycopg 3 +
pgvector · Alembic · httpx · Langfuse · uvicorn. Dependency management and the virtualenv are owned
by **uv** (`uv.lock` is committed and is the single source of truth).

The code lives in a single importable **`app/` package** at the repository root (no `src/`), split
by concern into `api/v1`, `core`, `models`, `repositories` and `schemas`. Alembic owns the schema
(`alembic/`), and an integration test suite in `tests/` runs against a real pgvector container.

> **This is a proof of concept and is expected to change significantly.** The API surface, storage
> schema, and passage-splitting strategy are all in flux. Treat the gaps listed at the end of this document
> as known PoC rough edges to be aware of, not as a standing work queue — do not "fix" them
> unprompted, and do not assume any current shape will survive.

## Repository Layout

```
app/main.py                 create_app() factory + module-level `app`; owns GET /health
app/config.py               pydantic-settings Settings class; module-level `settings` singleton
app/helper.py               get_app_version() — importlib.metadata, else read pyproject.toml
app/api/v1/router.py        Aggregates the search, ingest and notes routers; applies API-key auth
app/api/v1/ingest.py        POST /api/v1/ingest — split, embed in batches, write note + passages
app/api/v1/notes.py         GET / DELETE /api/v1/notes/{id}
app/api/v1/search.py        POST /api/v1/search — embed query, hybrid search, degrade to FTS-only
app/core/db.py              Lazy SQLAlchemy engine and session dependency
app/core/embeddings.py      get_embeddings() (batched), get_query_embedding() — Ollama or in-process
app/core/embedding_validation.py  Validates vectors before they are stored or searched with
app/core/splitter.py        Markdown note -> passages (frontmatter, headings, size limits)
app/core/pipeline.py        Content hash, pipeline version, FTS / embedding-input composition
app/core/errors.py          Exception handlers mapping embedding and DB failures to 4xx/5xx
app/core/security.py        API-key lookup -> ApiKey (client_name)
app/core/body_limit.py      Request-body size limit
app/core/schema_checks.py   Startup check: EMBEDDING_DIMENSION vs passages.embedding
app/models/                 SQLAlchemy models: Note, Passage, ApiKey
app/repositories/note.py    Tenant-scoped note and passage reads/writes
app/repositories/passage_search.py  The one-statement hybrid RRF query
app/repositories/search_guards.py   Per-transaction hnsw / statement_timeout settings, read back
app/schemas/ingest.py       IngestRequest, IngestResponse
app/schemas/note.py         NoteResponse
app/schemas/search.py       SearchRequest, SearchResult, SearchResponse
app/schemas/pagination.py   PaginationResponse — unused by design, see Known Gaps
alembic/                    Migrations; the head revision defines the schema
scripts/                    create_api_key, reingest, eval_queries, check_embedding_dimension, ...
tests/                      Integration suite (testcontainers pgvector, deterministic fake embeddings)
export_openapi.py           Regenerates openapi.json + openapi.yaml from the live app
openapi.json / openapi.yaml Committed API contract; drift-checked in CI
.spectral.yaml              Spectral ruleset used by the OpenAPI lint step
README.md                   Primary user-facing docs: setup, API examples, configuration, limitations
CHANGELOG.md                Generated by release-please — never edited by hand
docs/                       Design specs (docs/superpowers/specs) and audit reports — frozen records
LICENSE                     MIT
pyproject.toml              Metadata + runtime deps + [dependency-groups] dev
                            + [project.optional-dependencies] local-embeddings
                            No [build-system] — a virtual uv project, not a published wheel
uv.lock                     Locked dependency graph — committed on purpose, never hand-edited
.env.example                Documented configuration template
release-please-config.json  Release automation config
.release-please-manifest.json  Current released version
.github/workflows/          ci.yml, openapi.yml, release-please.yml
.github/dependabot.yml      Weekly update PRs for the uv and github-actions ecosystems
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

# Create / upgrade the schema (needs DATABASE_URL)
uv run alembic upgrade head

# Run the integration suite (needs Docker)
uv run --locked --no-build pytest -q

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
| PostgreSQL + pgvector >= 0.8 | storage                | `alembic upgrade head` creates the extension, tables and indexes |
| Ollama               | embeddings                     | model + `EMBEDDING_DIMENSION` must agree (0.6b=1024, 4b=2560, 8b=4096) |
| Langfuse             | tracing                        | optional; disabled when keys are empty                       |

## Database Schema

Alembic owns the schema: `uv run alembic upgrade head`. Do not create tables by hand; the models
and migrations are checked against each other (`tests/test_schema_drift.py`). In outline:

- `notes` — one row per `(client_name, scope, path_or_uri)`: frontmatter fields, `content_hash`,
  `pipeline_version`, `body`, `meta`, timestamps.
- `passages` — the splits of a note: `client_name`, `scope`, `ordinal`, `heading_path`, `content`,
  `tsv` (FTS, GIN-indexed), `embedding vector(N)` (HNSW-indexed, `N == EMBEDDING_DIMENSION`) and
  `embedding_degenerate`, a stored generated flag for vectors whose self-distance is NaN.
- `api_keys` — hashed API keys and their `client_name`.

The authoritative definitions are `alembic/versions/` and `app/models/`; the design rationale is in
`docs/superpowers/specs/2026-10-29-notes-passages-slice-1-design.md`. Every query carries a
`client_name` predicate. Search is `app/repositories/passage_search.py`: one statement with a
keyword arm and a vector arm, each filtered by client and scope before its own `LIMIT`, fused by
RRF. Repository code sets per-transaction GUCs with `set_config(..., true)` and reads them back,
raising if they did not take effect.

## Configuration

All settings live in `app/config.py` and are read from `.env` (`extra="ignore"`, so unknown keys are
tolerated). The `settings` object is instantiated **at import time**.

| Setting               | Default                                      | Used by                     |
| --------------------- | -------------------------------------------- | --------------------------- |
| `database_url`        | *(required)*                                 | the SQLAlchemy engine in `app/core/db.py` |
| `statement_timeout_ms` / `db_connect_timeout_s` | `5000` / `10`        | validated ranges, see `app/config.py` |
| `embedding_backend`   | `ollama`                                     | embeddings — `Literal["ollama", "local"]`, so a typo fails at startup |
| `local_embedding_model` | `sentence-transformers/all-MiniLM-L6-v2`   | only when the backend is `local` (384-wide, unlike the 1024 column) |
| `ollama_url`          | `http://localhost:11434/api/embed`           | `ollama` backend; must be the batch `/api/embed` endpoint — the legacy `/api/embeddings` is rejected at startup |
| `ollama_model`        | `qwen3-embedding:0.6b`                        | embeddings, `/health` |
| `embedding_dimension` | `1024`                                       | checked against `passages.embedding` at startup (`app/core/schema_checks.py`) |
| `search_candidates`   | `50`                                         | candidates per search arm before fusion (1..1000) |
| `rrf_k`               | `60`                                         | RRF damping constant (>= 1) |
| `fts_weight` / `vector_weight` | `1.0` / `1.0`                       | per-arm multipliers; 0 switches an arm off, both 0 is rejected |
| `query_instruction`   | Qwen3 retrieval instruction                  | prefix for **query** embeddings only; empty disables |
| `query_embed_timeout_s` | `5`                                        | query-embedding timeout (0 < x <= 30); on transient failure the search degrades |
| `embed_batch_size`    | `16`                                         | passages per `/api/embed` call at ingest (1..256) |
| `langfuse_public_key` / `_secret_key` / `_host` | empty / `https://cloud.langfuse.com` | *not read in code* — see Known Gaps |
| `app_name`            | `Nescio Semantic Memory API`                  | `FastAPI(title=...)`        |
| `log_level`           | `INFO`                                       | *not read in code*          |
| `host` / `port`       | `localhost` / `8080`                          | *not read in code* — no `uvicorn.run()` block exists |

Removed in 0.6.0: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_CONTEXT_CHARS`, `MAX_CONTEXT_DOCUMENTS`. They
are harmless in an old `.env` (unknown keys are ignored). When you add, remove or change a setting,
update `.env.example`, the README configuration table and — for a breaking change — the README's
"Upgrading to" note.

Changing the model or width **requires** a migration of `passages.embedding` and a re-ingest.

## CI, Releases, and Commit Conventions

Three GitHub Actions workflows:

- **`ci.yml`** — on push to `main` and all PRs. Runs `uv sync --locked` on Python **3.12, 3.13 and 3.14**
  (floor + current), then an import check of the runtime deps, then
  `uv sync --locked --extra local-embeddings --dry-run` so the opt-in extra stays resolvable without
  anyone paying for a torch download. It also runs the pytest suite, which needs Docker for the pgvector testcontainer. This workflow is the guard against `uv.lock` drift: **adding or changing a
  dependency without re-running `uv lock` breaks CI.**
- **`openapi.yml`** — on push to `main` and all PRs. Syncs the **dev** dependency group, regenerates
  the spec, validates it with `openapi-spec-validator`, fails if `git diff` shows
  `openapi.json`/`openapi.yaml` changed, then lints with Spectral using `.spectral.yaml`.
  **Any edit to routes, `Form(...)` fields, or Pydantic models must be followed by
  `uv run python export_openapi.py` and committing the regenerated files.**
  Both spec files are also release-please `extra-files`, and its updaters re-serialize the whole
  document rather than patching one line — so `export_openapi.py` deliberately reproduces their
  formatting via an indented-sequence YAML dumper and by writing integral floats as ints. **Do not
  simplify either away**: the drift check then fails on every release commit, which is how v0.2.1's
  release PR was caught.
- **Both workflows must invoke Python tools through `uv run`.** `uv sync` populates `.venv` but does
  not activate it (setup-uv's `activate-environment` defaults to false), so a bare `python` or
  console script resolves to the runner's own interpreter and fails on import.
- **`release-please.yml`** — on push to `main`. Maintains a release PR from conventional commits;
  merging it tags the release. `bump-minor-pre-major: true` means that in 0.x a breaking or feature
  change bumps the **minor** instead of the major — a plain `fix:` still bumps the patch, which is
  why an early splitter fix produced 0.2.1 rather than 0.3.0. Three
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
- **`.github/dependabot.yml`** — weekly update PRs for two ecosystems: `uv`, which bumps
  `pyproject.toml` and `uv.lock` together so `uv sync --locked` stays green, and `github-actions`.
  Commit prefixes are `build` and `ci`, neither of which triggers a release-please bump, so
  dependency updates never cut a release on their own. Minor and patch Python bumps are grouped
  into one PR; majors stay individual so each gets read.

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
- **Dev-only tooling** goes in `[dependency-groups] dev` (currently `openapi-spec-validator`, `pytest`, `testcontainers`), not
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
- **Workflow actions are pinned to commit SHAs**, not tags, with the release version in a trailing
  comment (`actions/checkout@<40-hex>  # v7.0.1`). A moved or compromised tag then cannot redirect a
  workflow, and the `github-actions` Dependabot ecosystem updates the SHA and that comment together,
  so the pins do not rot. Do not shorten a pin back to a tag: `astral-sh/setup-uv` stopped
  publishing moving major tags in v8, so `@v10` does not even resolve. The pin on
  `release-please-action` matters most — it is the one action handed a write-scoped PAT.
- **Code style:** type hints on public helpers and Pydantic models, `X | None` union syntax,
  SQLAlchemy with bound parameters (no string interpolation of user input), and a `client_name`
  predicate in every query. One concern per module: endpoints in `app/api/v1/`, infrastructure in
  `app/core/`, queries in `app/repositories/`, ORM models in `app/models/`, request/response models
  in `app/schemas/`. Comments explain why and cite issue numbers; keep files under 400 lines. Handlers are decorated with `@observe(name="...")` for Langfuse tracing, and
  endpoints declare `summary=`/`description=` plus `tags=[...]` so the generated spec passes the
  Spectral lint step. The old `""" ... """` section banners are gone — they belonged to the
  single-file layout.
- **Tests:** `uv run --locked --no-build pytest -q` (needs Docker). The `client` fixture in
  `tests/conftest.py` patches the embedding functions with the deterministic fakes in
  `tests/fakes.py`; the `fts_only` fixture sets `vector_weight=0`. Cross-cutting behaviour (auth,
  error mapping, statement timeout, transaction guard) is tested against the passages API.
- **Docs:** `README.md` is the user-facing contract (setup, API examples, configuration,
  limitations); this file is the agent-facing one. When behaviour changes, update the README too —
  especially the configuration table, the "Upgrading to" note for breaking changes, and the "Current
  limitations" list. `CHANGELOG.md` is generated by release-please, and `docs/superpowers/*` and
  `docs/reports/*` are dated records: do not edit them.
- **Licensing:** MIT. New files need no license header, but any vendored third-party code must be
  MIT-compatible and attributed.

## Known Gaps and Gotchas

Verified against the current code. These are PoC-stage rough edges, **not** a work queue — treat
them as context so you don't build on a false assumption, and don't fix them unprompted. The
user-facing subset is published in the README under "Current limitations", so if you *do* fix one,
update both places.

1. **`app/helper.py` holds the only version-sensitive import.** `tomllib` is stdlib only since
   Python **3.11** and is imported unconditionally at module scope, so lowering `requires-python`
   below 3.11 would fail with `ModuleNotFoundError` at import time — *outside* the function's own
   `except Exception` guard, which cannot catch it. Supporting an older floor needs
   `tomli>=2.0; python_version < "3.11"` plus a guarded import. Inside the current `>=3.12` floor
   this is a non-issue, and the comment there now says so accurately. (`importlib.metadata`, also
   used in that module, has been stdlib since 3.8.)
2. **The `local` embeddings backend has never run against the real library.**
   `sentence-transformers` now lives in the `local-embeddings` optional extra and `get_embedding()`
   branches on `settings.embedding_backend`, but that branch has only been exercised with a stubbed
   `SentenceTransformer`. Nothing in CI installs the extra — deliberately, since torch is ~530 MB of
   Linux wheels — so a change to the real `.encode()` signature or its return dtype would go
   unnoticed until someone opts in. Verify against the real library before relying on it.
3. **Langfuse keys are not wired.** `@observe` relies on the SDK picking up `LANGFUSE_*` from the
   process environment; `settings.langfuse_*` are never passed to the SDK, and pydantic-settings
   does **not** export `.env` values into `os.environ`. Tracing from a `.env`-only setup will not
   attach.
4. **Unmapped errors are still bare 500s.** `app/core/errors.py` maps embedding-backend, operational
   and data errors to documented responses, but there is no catch-all `Exception` handler. Ingest
   embeds in batches of `EMBED_BATCH_SIZE` with no retry or backoff.
5. **`settings.host` / `port` / `log_level` are unused.** The server must be started with an
   explicit `uvicorn` command line. (`embedding_dimension` is *not* unused: it is checked against
   `passages.embedding` at startup.)

**Not a gap:** `app/schemas/pagination.py` (`PaginationResponse`) has no importers and looks like
dead code, but it is deliberate scaffolding for paged retrieval — downstream
agent clients are expected to consume bounded result sets rather than one large batch. Do **not**
remove it during cleanup or simplification passes. Its camelCase aliases (`totalCount`,
`pageSize`) differ from `IngestResponse`'s short-form aliases (`file`, `ingested`) and from
`SearchResult`, which has none; reconcile that deliberately when pagination is wired in.

### Repo hygiene

Dated `*.txt` files at the repository root (`YYYY-MM-DD-HHMMSS-<slug>.txt`) are exported
AI-session transcripts. They are ignored scratch artifacts — do **not** commit them, and never
use broad staging like `git add -A` here. Stage explicit paths only.
