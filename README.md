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
- **PostgreSQL 16+** with the `pgvector` extension
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

Liveness probe. Reports the active embedding backend and model, and deliberately does **not**
contact PostgreSQL or Ollama — so an unhealthy dependency cannot make Kubernetes restart the pod.

```bash
curl http://localhost:8000/health
# {"status":"ok","embedding_backend":"ollama","embedding_model":"qwen3-embedding:0.6b"}
```

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
| `DATABASE_URL`        | *required*                                 | PostgreSQL connection string                         |
| `EMBEDDING_BACKEND`   | `ollama`                                    | `ollama` (HTTP) or `local` (in-process); any other value fails at startup |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2`  | model for the `local` backend only — see [Embedding backends](#embedding-backends) |
| `OLLAMA_URL`          | `http://localhost:11434/api/embeddings`     | embedding endpoint (local or remote)                 |
| `OLLAMA_MODEL`        | `qwen3-embedding:0.6b`                      | embedding model name                                 |
| `EMBEDDING_DIMENSION` | `384`                                       | must match the model **and** the `vector(N)` column  |
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

The default `LOCAL_EMBEDDING_MODEL` (`all-MiniLM-L6-v2`) is 384-dimensional, the same width as the
default `qwen3-embedding:0.6b`, so either backend works against a `vector(384)` column. Choosing a
model of a different width means recreating the column and re-ingesting.

### Ollama models

| Model                   | Dimensions | Rough cost          |
| ----------------------- | ---------- | ------------------- |
| `qwen3-embedding:0.6b`  | 384        | ~400 MB RAM         |
| `qwen3-embedding:4b`    | 1024       | ~2.5 GB RAM         |
| `qwen3-embedding:8b`    | 4096       | GPU recommended     |

Switching models means recreating the `embedding` column at the new dimension and re-ingesting
everything — vectors of different dimensions are not comparable.

## Development

```bash
uv sync --locked                      # install / refresh the environment
uv sync --locked --extra local-embeddings  # only if using the in-process backend
uv lock                               # after changing dependencies in pyproject.toml
uv run python export_openapi.py       # regenerate openapi.json + openapi.yaml
uv run uvicorn app.main:app --reload  # dev server (from the repository root)
uv run pytest                         # integration suite (needs Docker)
uv run alembic upgrade head           # apply migrations
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

- **`EMBEDDING_DIMENSION` is never checked.** It is documentation only; a mismatch with the actual
  `vector(N)` column surfaces as a database error on insert rather than at startup.
- **Langfuse keys in `.env` are ignored.** `@observe` relies on the SDK reading `LANGFUSE_*` from
  the process environment, and pydantic-settings does not export `.env` values into it. Export the
  keys in your shell or service manager, or tracing silently stays off.
- **Unmapped errors still return a bare `500`.** Dependency failures now return a
  documented `503` or `500` with a `{"detail": ...}` body, but any exception outside
  those handlers surfaces as plain-text `Internal Server Error` rather than JSON, so
  a client cannot rely on parsing `detail` from every error.
- **Ingestion is serial and chatty.** One blocking Ollama call per chunk, with no batching and no
  retry/backoff.
- **No index on `(client_name, repo_name, file_path)` as a unit.** Each column is indexed
  separately, so the delete-on-re-ingest path relies on PostgreSQL combining them rather than a
  single composite index.

Contributions addressing any of the above are welcome.

## License

MIT — see [LICENSE](LICENSE).
