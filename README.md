# nescio-memory

A small semantic-memory API. Feed it text, and it chunks it, embeds it, and stores it in
PostgreSQL + pgvector so you can later retrieve the most relevant passages by meaning rather than
by keyword.

Designed as the long-term memory layer for agents and coding assistants: `/api/v1/ingest` what a
project or document says, `/api/v1/search` it back when you need context.

> **Status: proof of concept.** The API surface, storage schema, and chunking strategy are all
> expected to change significantly — see [Current limitations](#current-limitations).

## How it works

```
                              ┌──────────────────────┐
  POST /api/v1/ingest ───────▶│  chunk_text()        │  sliding window + overlap
                              │  CHUNK_SIZE/OVERLAP  │  (default 1000 / 200)
                              └──────────┬───────────┘
                                         │ one call per chunk
                                         ▼
                              ┌──────────────────────┐
                              │  Ollama embeddings   │  HTTP, not in-process
                              └──────────┬───────────┘
                                         ▼
                              ┌──────────────────────┐
                              │  PostgreSQL/pgvector │  table: learnings
                              └──────────▲───────────┘
                                         │ ORDER BY embedding <=> query  (cosine distance)
  POST /api/v1/search ───────────────────┘
```

Embeddings are produced by an external [Ollama](https://ollama.com) server, so the API process
stays small and the model can run wherever you have capacity — on Kubernetes that is typically a
separate Ollama pod behind a Service. An optional in-process backend using
[sentence-transformers](https://sbert.net) is available for deployments that cannot run a second
service; see [Embedding backends](#embedding-backends). Both endpoints are traced with
[Langfuse](https://langfuse.com) when `LANGFUSE_*` are exported into the **process environment** —
values that only live in `.env` are not picked up, see
[Current limitations](#current-limitations).

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — manages the virtualenv and the lockfile
- **PostgreSQL 16+** with the `pgvector` extension, **`pgvector` >= 0.8.0 required** — search sets
  `hnsw.iterative_scan`, which earlier versions reject outright
- **Docker** — only to run the test suite, which starts a real PostgreSQL + pgvector container
- **Ollama** running an embedding model (e.g. `qwen3-embedding`) — unless you use the optional
  in-process backend instead

## Quick start

### 1. Install dependencies

```bash
uv sync --locked
```

Only if you want the in-process embedding backend instead of Ollama, add the optional extra —
this pulls in torch, roughly 530 MB of wheels on Linux:

```bash
uv sync --locked --extra local-embeddings
```

### 2. Configure

```bash
cp .env.example .env
```

Then edit `.env`. The only **required** value is `DATABASE_URL` — the app fails to start without
it. Make sure `OLLAMA_MODEL` and `EMBEDDING_DIMENSION` agree with each other *and* with the
embedding model you choose.

### 3. Create the database schema

Alembic owns the schema. From the repository root, with `DATABASE_URL` already set (see step 2):

```bash
uv run alembic upgrade head
```

This creates the `vector` extension, the `learnings` and `api_keys` tables, and every index,
including the HNSW index used for similarity search. Do not create tables by hand — the models and
the migrations are checked against each other, and a hand-built schema will be missing columns the
application requires.

The application also verifies at startup that `EMBEDDING_DIMENSION` matches the
`learnings.embedding` column, so it will not start against an unmigrated or
mismatched database. That means a reachable database is required to boot.

### 4. Create an API key

Every `/api/v1` route requires an API key. Mint one per client:

```bash
uv run python scripts/create_api_key.py my-client
```

```
client_name : my-client
api_key     : nm_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
⚠️  Store this now — it cannot be retrieved again.
```

Only a SHA-256 hash of the key is stored, so the plaintext cannot be recovered — mint a new key if
you lose it. Omit the argument and the client is named `default`.

The `client_name` is not just a label: every ingested row is tagged with it, and each client sees
only its own data. See [Authentication](#authentication).

### 5. Run

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Run this **from the repository root**: `app` is imported from the working directory, and `.env` is
resolved relative to it as well.

Interactive docs are served by FastAPI at <http://localhost:8000/docs>. The committed contract
lives in [`openapi.json`](openapi.json) / [`openapi.yaml`](openapi.yaml).

## API

### `GET /health`

Liveness probe. Reports the active embedding backend, model and configured vector width, and
deliberately does **not** contact PostgreSQL or Ollama — so an unhealthy dependency cannot make
Kubernetes restart the pod.

```bash
curl http://localhost:8000/health
# {"status":"ok","embedding_backend":"ollama","embedding_model":"qwen3-embedding:0.6b",
#  "embedding_dimension":1024}
```

`embedding_dimension` is the **configured** width, not a measured one — this endpoint contacts
nothing. To check it against what the model really emits, see
[Checking the dimension](#checking-the-dimension).

### Authentication

Every route under `/api/v1` requires an `X-API-Key` header. `GET /health` does not, so a liveness
probe needs no credentials.

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: nm_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "top_k": 3}'
```

| Condition                  | Response                                            |
| -------------------------- | --------------------------------------------------- |
| header absent              | `401 {"detail":"Missing API key"}`                   |
| key unknown or revoked     | `401 {"detail":"Invalid or revoked API key"}`        |

Keys are rows in `api_keys`, not configuration — there is no `API_KEY` environment variable. Only a
SHA-256 hash and a short display prefix are stored, so a database leak yields nothing usable. A
fast hash is deliberate: the keys are 32 bytes of `secrets.token_urlsafe` entropy, so there is no
dictionary to stretch against and a per-request KDF would only add latency.

Revoke a key by setting `revoked_at` on its row rather than deleting it, which keeps the audit
trail:

```sql
UPDATE api_keys SET revoked_at = now() WHERE key_prefix = 'nm_xxxxxxxxxxx';
```

#### Per-client isolation

The key does more than authenticate. Every row written through it is tagged with its `client_name`,
and that tag scopes every read and delete:

- Search returns only the calling client's rows.
- Re-ingesting a file deletes only that client's previous chunks for it.
- Two clients can hold the same `repo_name` and `file_path` without seeing or overwriting each
  other's copy.

### `POST /api/v1/ingest`

Stores a file's content. Accepts `application/x-www-form-urlencoded` (or `multipart/form-data`)
with three fields: `repo_name`, `file_path`, `content`.

Existing chunks for the same `(client_name, repo_name, file_path)` triple are deleted first, so
re-ingesting an updated file is safe and idempotent — and scoped to the calling client, so it never
touches another client's copy of the same path. Chunks shorter than 50 characters are dropped.
Content over 500,000 characters is rejected with a `400`.

**That 500,000-character cap is a character limit, not a byte limit — and for non-ASCII content a
smaller byte limit fires first.** The form parser (Starlette, underneath FastAPI's `Form(...)`)
enforces its own 1,048,576-byte (1024KB) limit per form field, ahead of and independent from the
500,000-character check above. ASCII content hits both limits at the same size, but multi-byte
content hits the byte limit at a much lower character count — e.g. 3-byte-per-character CJK text is
capped at roughly 349,000 characters, well under 500,000. When the byte limit fires first, the
response is still a `400`, but with Starlette's own message (`"Field exceeded maximum size of
1024KB."`) rather than the character-count message above. There is no supported way to raise that
byte limit for a `Form(...)`-declared route in the installed FastAPI/Starlette versions without
changing this endpoint's request schema, so this is documented behavior rather than a bug fix.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-API-Key: nm_your_key_here" \
  -F "repo_name=nescio-memory" \
  -F "file_path=docs/architecture.md" \
  -F "content=Embeddings are produced by an external Ollama server..."
```

```json
{
  "status": "success",
  "file": "docs/architecture.md",
  "ingested": 3
}
```

| Field      | Description                                        |
| ---------- | -------------------------------------------------- |
| `status`   | `"success"`                                        |
| `file`     | the `file_path` that was ingested                  |
| `ingested` | number of chunks actually stored (after filtering) |

### `POST /api/v1/search`

Returns the `top_k` most similar chunks, scored as cosine similarity (`1.0` = identical).

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: nm_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{"query": "where do embeddings come from?", "top_k": 3}'
```

```json
{
  "results": [
    {
      "content": "Embeddings are produced by an external Ollama server...",
      "metadata": {
        "file_name": "architecture.md",
        "relative_path": "docs/architecture.md",
        "chunk_index": 0
      },
      "similarity": 0.8123
    }
  ]
}
```

| Field         | Type             | Default | Description                            |
| ------------- | ---------------- | ------- | -------------------------------------- |
| `query`       | string           | —       | text to search for (required)          |
| `top_k`       | int              | `5`     | maximum number of results; must be between 1 and 50, otherwise `422` |
| `repo_filter` | string \| `null` | `null`  | restrict results to one `repo_name`    |

Each result carries `content` (the matched chunk), `metadata` (`file_name`, `relative_path`,
`chunk_index`) and `similarity`.

## Configuration

All settings are read from `.env` via [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/);
see [`.env.example`](.env.example) for the annotated template. Unknown keys are ignored.

| Variable              | Default                                    | Description                                          |
| --------------------- | ------------------------------------------ | ---------------------------------------------------- |
| `DATABASE_URL`        | *required*                                 | PostgreSQL connection string; a plain `postgresql://` uses the bundled psycopg 3 driver |
| `STATEMENT_TIMEOUT_MS` | `5000`                                     | caps the vector-search statement only (not the whole request); must be `> 0`, otherwise the app refuses to start |
| `EMBEDDING_BACKEND`   | `ollama`                                    | `ollama` (HTTP) or `local` (in-process); any other value fails at startup |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2`  | model for the `local` backend only — see [Embedding backends](#embedding-backends) |
| `OLLAMA_URL`          | `http://localhost:11434/api/embeddings`     | embedding endpoint (local or remote)                 |
| `OLLAMA_MODEL`        | `qwen3-embedding:0.6b`                      | embedding model name                                 |
| `EMBEDDING_DIMENSION` | `1024`                                      | must match the model **and** the `vector(N)` column — the default is the width `qwen3-embedding:0.6b` actually emits; verify with [`scripts/check_embedding_dimension.py`](scripts/check_embedding_dimension.py) |
| `LANGFUSE_PUBLIC_KEY` | *(empty → tracing off)*                     | Langfuse credentials                                 |
| `LANGFUSE_SECRET_KEY` | *(empty → tracing off)*                     |                                                      |
| `LANGFUSE_HOST`       | `https://cloud.langfuse.com`                | Langfuse instance                                    |
| `APP_NAME`            | `Nescio Semantic Memory API`                | title shown in the OpenAPI docs                      |
| `LOG_LEVEL`           | `INFO`                                      | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` \| `CRITICAL` |
| `CHUNK_SIZE`          | `1000`                                      | characters per chunk                                 |
| `CHUNK_OVERLAP`       | `200`                                      | overlap between chunks — must satisfy `0 <= CHUNK_OVERLAP < CHUNK_SIZE`, otherwise the app refuses to start |
| `HOST` / `PORT`       | `localhost` / `8080`                        | *currently not applied* — pass these to `uvicorn` instead |

### Embedding backends

|                  | `ollama` (default)                    | `local`                                        |
| ---------------- | ------------------------------------- | ---------------------------------------------- |
| Runs             | in a separate Ollama process or pod   | in-process, via sentence-transformers          |
| Install          | nothing extra                         | `uv sync --extra local-embeddings` (adds torch) |
| Model weights    | managed by Ollama                     | fetched from Hugging Face Hub on first use     |
| Scales           | independently of the API              | with the API process — needs CPU/GPU and RAM there |
| Best for         | clusters that already run Ollama      | single-process or offline deployments          |

Select the backend with `EMBEDDING_BACKEND`. The `local` backend loads its model once and caches
it for the lifetime of the process. Selecting `local` without installing the extra fails on the
first embedding call with an error naming the install command — it does **not** silently fall back
to Ollama.

#### The two backends are not interchangeable against one database

The default `LOCAL_EMBEDDING_MODEL` (`all-MiniLM-L6-v2`) is **384**-dimensional. The default
`OLLAMA_MODEL` (`qwen3-embedding:0.6b`) is **1024**-dimensional. A database has one
`vector(N)` column, so it serves one of them, and switching backends is a migration plus a
re-ingest — not a configuration flip. (An earlier version of this README claimed both were
384-wide and therefore interchangeable. They never were; see
[#36](https://github.com/noctua84/nescio-memory/issues/36).)

| Backend                   | Default model                 | Width    | Schema it needs                                  |
| ------------------------- | ----------------------------- | -------- | ------------------------------------------------ |
| `ollama` (default)        | `qwen3-embedding:0.6b`        | **1024** | migration head — `EMBEDDING_DIMENSION=1024`      |
| `local`                   | `all-MiniLM-L6-v2`            | **384**  | stop at `3d80eb22a17d` — `EMBEDDING_DIMENSION=384` |

Running the `local` backend therefore means **both** of:

```bash
EMBEDDING_DIMENSION=384                   # in .env
uv run alembic upgrade 3d80eb22a17d       # NOT `head` — head is vector(1024)
```

`3d80eb22a17d` is the last revision before the column widened to 1024, and
`alembic downgrade -1` from head lands there. With the stock `EMBEDDING_DIMENSION=1024` the
`local` backend fails every embedding call, loudly, on the dimension check — the mirror image of
the defect in #36.

Pointing `LOCAL_EMBEDDING_MODEL` at a genuinely 1024-dimensional sentence-transformers model
would make one schema serve both, and is the obvious next step; it is not taken here because
nobody has run it. An unexecuted dimension claim in this file is what caused #36 in the first
place.

### Ollama models

| Model                   | Dimensions | Rough cost          |
| ----------------------- | ---------- | ------------------- |
| `qwen3-embedding:0.6b`  | 1024       | ~400 MB RAM         |
| `qwen3-embedding:4b`    | 2560       | ~2.5 GB RAM         |
| `qwen3-embedding:8b`    | 4096       | GPU recommended     |

Widths taken from the models' own `hidden_size` (`Qwen/Qwen3-Embedding-{0.6B,4B,8B}`), and 0.6b
confirmed against a running server: `/api/show` reports `qwen3.embedding_length: 1024` and
`/api/embeddings` returns 1024 floats. The previous version of this table said 384 and 1024 for
the first two rows, which is where #36 came from — do not re-derive a width from this file
without running `python -m scripts.check_embedding_dimension` against the model.

Switching models means recreating the `embedding` column at the new dimension and re-ingesting
everything — vectors of different dimensions are not comparable.

### Checking the dimension

`EMBEDDING_DIMENSION` has to agree with two independent things, and until #36 only one of them
was ever checked:

| Must agree with                       | Checked by                                                  | When            |
| ------------------------------------- | ----------------------------------------------------------- | --------------- |
| the `learnings.embedding` column      | `verify_embedding_dimension` (`app/core/schema_checks.py`)   | every startup   |
| the SQLAlchemy model                  | `tests/test_schema_drift.py`                                | every CI run    |
| **what the model actually emits**     | `scripts/check_embedding_dimension.py`                      | **on demand**   |

The third row is the gap that made the stock configuration inert while the suite stayed green: the
tests embed through a fake that returns `EMBEDDING_DIMENSION` floats by construction, so they agree
with the setting no matter what it says. Run the script after changing `OLLAMA_MODEL`,
`LOCAL_EMBEDDING_MODEL`, `EMBEDDING_BACKEND` or `EMBEDDING_DIMENSION`:

```bash
uv run python -m scripts.check_embedding_dimension
# OK  ollama backend at http://localhost:11434/api/embeddings, model qwen3-embedding:0.6b
#   emits 1024 dimensions, matching EMBEDDING_DIMENSION
```

It exits `0` on a match, `1` on a mismatch and `2` if the backend could not be reached, so a
deployment pipeline can treat "wrong" and "could not tell" differently.

This is a script and not a startup check on purpose: probing at boot would make the API's liveness
depend on the embedding backend, and an API that will not start because Ollama is restarting is a
worse failure than the one being guarded against. `tests/test_embedding_width_contract.py` runs the
same probe in the suite and **skips** when no backend is reachable — the skip is printed in every
run's summary (`addopts = "-ra"`), because a silent skip would leave the same hole.

## Development

```bash
uv sync --locked                      # install / refresh the environment
uv sync --locked --extra local-embeddings  # only if using the in-process backend
uv lock                               # after changing dependencies in pyproject.toml
uv run python export_openapi.py       # regenerate openapi.json + openapi.yaml
uv run uvicorn app.main:app --reload  # dev server (from the repository root)
uv run pytest                         # integration suite (needs Docker)
uv run alembic upgrade head           # apply migrations
uv run python -m scripts.check_embedding_dimension  # does the model's width match EMBEDDING_DIMENSION?
```

Four things are enforced by CI:

- **The test suite must pass.** `uv run pytest` runs 30 integration tests against a real PostgreSQL
  + pgvector container on every push and pull request.
- **`uv.lock` must stay in sync with `pyproject.toml`.** If you add or change a dependency, run
  `uv lock` and commit both files.
- **The committed OpenAPI spec must match the code.** If you touch a route, a `Form(...)` field, or
  a Pydantic model, run `export_openapi.py` and commit the regenerated `openapi.json` and
  `openapi.yaml`.
- **The `local-embeddings` extra must still resolve.** A `--dry-run` sync verifies it against the
  lockfile on every run without downloading torch, so the opt-in backend cannot rot unnoticed.

Releases are automated by [release-please](https://github.com/googleapis/release-please) from
Conventional Commits (`feat:`, `fix:`, `chore:`, …). Merging the release PR it maintains bumps the
version and publishes a GitHub Release — do not edit versions by hand.

## Current limitations

This is a PoC. Known rough edges, roughly in order of how much they matter:

- **Search recall is bounded, not guaranteed.** pgvector's HNSW index is approximate and filters by
  `client_name` after producing candidates. The search now sets `hnsw.iterative_scan` so the index
  keeps looking until it has `top_k` matches: on a corpus where one client held 10 rows of 8,010,
  that took the result from 1 row to the full 10. It remains approximate — HNSW's graph construction
  is randomized and part of the graph can be unreachable for any given query, so a query can still
  come back one row short, and a client holding a very small share of a very large table can still
  receive a short result set rather than an error. `hnsw.max_scan_tuples` is a safety cap on work,
  not the limit on recall; raising it does not lengthen a short result.
- **Langfuse keys in `.env` are ignored.** `@observe` relies on the SDK reading `LANGFUSE_*` from
  the process environment, and pydantic-settings does not export `.env` values into it. Export the
  keys in your shell or service manager, or tracing silently stays off.
- **Unmapped errors still return a bare `500`.** `EmbeddingBackendError`, `OperationalError`,
  and `DataError` are mapped, so the common failure families now return a documented `503`,
  `500`, or `400` with a `{"detail": ...}` body. But there is no catch-all handler for
  `Exception`, so any genuinely unanticipated failure still surfaces as plain-text
  `Internal Server Error` rather than JSON, and a client still cannot rely on parsing
  `detail` from every error.
- **Ingestion is serial and chatty.** One blocking Ollama call per chunk, with no batching and no
  retry/backoff.

Contributions addressing any of the above are welcome.

## License

MIT — see [LICENSE](LICENSE).
