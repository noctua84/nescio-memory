# QWEN.md — nescio-memory

Instructional context for AI agents working in this repository.

## Project Overview

**nescio-memory** is a small FastAPI service that provides *semantic memory* over a PostgreSQL +
pgvector database. It exposes two operations:

- **`POST /ingest`** — accepts a file's text content (form-encoded), splits it into overlapping
  character chunks, embeds each chunk, and upserts them into the `learnings` table (deleting that
  file's previous chunks first, so re-ingesting is idempotent).
- **`POST /search`** — embeds a query and returns the `top_k` nearest chunks by cosine similarity.

Embeddings are **not** computed in-process. They are requested over HTTP from an
[Ollama](https://ollama.com) `/api/embeddings` endpoint (default model `qwen3-embedding:0.6b`,
384 dims). Langfuse `@observe` decorators are applied to both handlers for tracing.

Stack: Python ≥ 3.12 · FastAPI · Pydantic v2 / pydantic-settings · psycopg2 + pgvector · httpx ·
Langfuse · uvicorn. Dependency management and the virtualenv are owned by **uv** (`uv.lock` is
committed and is the single source of truth).

The project is intentionally a **flat, single-module layout** — no `src/`, no packages, no
migrations, no tests yet.

> **This is a proof of concept and is expected to change significantly.** The API surface, storage
> schema, and chunking strategy are all in flux. Treat the gaps listed at the end of this document
> as known PoC rough edges to be aware of, not as a standing work queue — do not "fix" them
> unprompted, and do not assume any current shape will survive.

## Repository Layout

```
main.py                     FastAPI app: helpers, Pydantic models, /search /ingest /health
config.py                   pydantic-settings Settings class; module-level `settings` singleton
export_openapi.py           Regenerates openapi.json + openapi.yaml from the live app
openapi.json / openapi.yaml Committed API contract; drift-checked in CI
README.md                   Primary user-facing docs: setup, SQL schema, API examples, limitations
LICENSE                     MIT
pyproject.toml              Metadata (description, readme, license, authors, urls) + dependencies
                            No [build-system] — this is a virtual uv project, not a published wheel
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

# Run the dev server
uv run uvicorn main:app --reload --port 8000

# Regenerate the OpenAPI spec after ANY route/model change (CI enforces this)
uv run python export_openapi.py

# Smoke-check the service
curl http://localhost:8000/health
```

**Critical gotcha:** `Settings.database_url` has **no default**, so importing `config` (and
therefore `main`) raises a `ValidationError` unless `DATABASE_URL` is present in `.env` or the
process environment. `export_openapi.py` needs it too — the OpenAPI workflow works around this by
injecting dummy `DATABASE_URL` / `OLLAMA_URL` env vars, because `app.openapi()` only reads route
definitions and never touches the DB. Use the same trick for any offline spec/docs generation.

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
  repo_name  text,          -- written as a column by /ingest
  file_path  text,          -- written as a column by /ingest
  content    text,          -- one chunk
  metadata   json/jsonb,    -- {"file_name", "relative_path", "chunk_index"}
  embedding  vector(N)      -- N == EMBEDDING_DIMENSION
)
```

`/search` orders by `embedding <=> %s` (cosine distance) and reports `1 - distance` as
`similarity`.

## Configuration

All settings live in `config.py` and are read from `.env` (`extra="ignore"`, so unknown keys are
tolerated). The `settings` object is instantiated **at import time**.

| Setting               | Default                                      | Used by                     |
| --------------------- | -------------------------------------------- | --------------------------- |
| `database_url`        | *(required)*                                 | `get_db_connection()`       |
| `ollama_url`          | `http://localhost:11434/api/embeddings`       | `get_embedding()`           |
| `ollama_model`        | `qwen3-embedding:0.6b`                        | `get_embedding()`, `/health` |
| `embedding_dimension` | `384`                                        | *not read in code* — must match the DB column |
| `langfuse_public_key` / `_secret_key` / `_host` | empty / `https://cloud.langfuse.com` | *not read in code* — see Known Gaps |
| `app_name`            | `Nescio Semantic Memory API`                  | `FastAPI(title=...)`        |
| `log_level`           | `INFO`                                       | *not read in code*          |
| `chunk_size` / `chunk_overlap` | `1000` / `200`                       | *not read in code* — `chunk_text()` uses its own hardcoded defaults |
| `host` / `port`       | `localhost` / `8080`                          | *not read in code* — no `uvicorn.run()` block exists |

Note the `.env.example` / `config.py` mismatch: the template ships `OLLAMA_MODEL=qwen3-embedding:4b`
with `EMBEDDING_DIMENSION=1024`, while the code defaults to `0.6b` / `384`. Changing the model
**requires** recreating the `embedding` column at the matching dimension.

## CI, Releases, and Commit Conventions

Three GitHub Actions workflows:

- **`ci.yml`** — on push to `main` and all PRs. Runs `uv sync --locked` on Python **3.12, 3.13 and 3.14**
  (floor + current), then an import check of the runtime deps. The test step is a deliberate `TODO`
  because pytest exits 5 on an empty suite. This workflow is the guard against `uv.lock` drift:
  **adding or changing a dependency without re-running `uv lock` breaks CI.**
- **`openapi.yml`** — on push to `main` and all PRs. Syncs the **dev** dependency group, regenerates
  the spec, validates it with `openapi-spec-validator`, fails if `git diff` shows
  `openapi.json`/`openapi.yaml` changed, then lints with Spectral using `.spectral.yaml`.
  **Any edit to routes, `Form(...)` fields, or Pydantic models must be followed by
  `uv run python export_openapi.py` and committing the regenerated files.**
- **Both workflows must invoke Python tools through `uv run`.** `uv sync` populates `.venv` but does
  not activate it (setup-uv's `activate-environment` defaults to false), so a bare `python` or
  console script resolves to the runner's own interpreter and fails on import.
- **`release-please.yml`** — on push to `main`. Maintains a release PR from conventional commits;
  merging it tags the release. `bump-minor-pre-major: true` (0.x releases bump the minor), and an
  `extra-files` entry rewrites the project's version **inside `uv.lock`** via jsonpath.
  `.release-please-manifest.json` and `pyproject.toml` versions must stay in sync. Requires the
  repo setting *Allow GitHub Actions to create and approve pull requests*.

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
- **Line endings:** `.gitattributes` stores LF and checks out native endings. `*.sh`, `*.yml`,
  `*.yaml`, and `uv.lock` are forced to LF because Linux tooling consumes them. Without this,
  Windows checkouts and the Ubuntu CI runner disagree and whole files appear modified. Do not
  delete or weaken these rules.
- **Secrets:** `.env` is gitignored; `.env.example` is the documented template. Never commit real
  keys.
- **Code style:** type hints on public helpers and Pydantic models, `X | None` union syntax,
  parameterized SQL via `%s` placeholders (no string interpolation of user input). The module uses
  `""" ... """` string literals as section banners (`Helper functions`, `Models`, `Endpoints`) —
  follow that when adding sections to `main.py`.
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

1. **`repo_filter` never matches.** `/search` filters on `metadata->>'repo_name'`, but `/ingest`
   writes `repo_name` only as a **table column** and stores just `file_name`, `relative_path`,
   `chunk_index` in `metadata`. Filtering by repo silently returns zero rows.
2. **`httpx` is imported but not declared.** `main.py` depends on it directly; it is only present
   transitively (via `langfuse` and `huggingface-hub`). Adding it to `pyproject.toml` is the safe fix.
3. **`sentence-transformers` is declared but never imported.** Embeddings come from Ollama over
   HTTP. It is a heavy dependency (pulls in torch), is currently pinned to `==6.1.0`, and is also
   one of the things keeping `httpx` in the lockfile — removing it affects gap #2.
4. **Chunking settings are inert.** `chunk_text()` hardcodes `chunk_size=1000, overlap=200`;
   `settings.chunk_size` / `chunk_overlap` are never passed in, so `CHUNK_SIZE` / `CHUNK_OVERLAP`
   env vars have no effect. The sliding-window loop also emits a final short chunk and skips chunks
   under 50 characters, so `chunks_ingested` can be lower than the chunk count.
5. **Langfuse keys are not wired.** `@observe` relies on the SDK picking up `LANGFUSE_*` from the
   process environment; `settings.langfuse_*` are never passed to the SDK, and pydantic-settings
   does **not** export `.env` values into `os.environ`. Tracing from a `.env`-only setup will not
   attach.
6. **No resilience or pooling.** A new `psycopg2` connection is opened per request and closed
   manually (no context managers, no `try/finally`), `HTTPException` is imported but unused, and
   there is no error handling around DB or Ollama failures — any error surfaces as a bare 500.
   `/ingest` also calls Ollama once **per chunk**, serially.
7. **`settings.host` / `port` / `log_level` are unused**; the server must be started with an
   explicit `uvicorn` command line.

### Repo hygiene

Dated `*.txt` files at the repository root (e.g.
`2026-09-18-140231-can-you-handle-the-commit-message-part-for-the-st.txt`) are exported
AI-session transcripts. They are untracked scratch artifacts — do **not** commit them, and never
use broad staging like `git add -A` here. Stage explicit paths only.
