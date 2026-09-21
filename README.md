# nescio-memory

A small semantic-memory API. Feed it text, and it chunks it, embeds it, and stores it in
PostgreSQL + pgvector so you can later retrieve the most relevant passages by meaning rather than
by keyword.

Designed as the long-term memory layer for agents and coding assistants: `/api/v1/ingest` what a
project or document says, `/api/v1/search` it back when you need context.

> **Status: proof of concept.** The API surface, storage schema, and chunking strategy are all
> expected to change significantly. There are no migrations and no tests yet — see
> [Current limitations](#current-limitations).

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
stays small and the model can run wherever you have capacity. Both endpoints are traced with
[Langfuse](https://langfuse.com) when `LANGFUSE_*` are exported into the **process environment** —
values that only live in `.env` are not picked up, see
[Current limitations](#current-limitations).

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — manages the virtualenv and the lockfile
- **PostgreSQL 16+** with the `pgvector` extension
- **Ollama** running an embedding model (e.g. `qwen3-embedding`)

## Quick start

### 1. Install dependencies

```bash
uv sync --locked
```

### 2. Create the database schema

There are no migration files yet, so create the table manually. Replace `1024` with the dimension
of your chosen embedding model (see the table in [Configuration](#configuration)):

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE learnings (
    id         bigserial PRIMARY KEY,
    repo_name  text        NOT NULL,
    file_path  text        NOT NULL,
    content    text        NOT NULL,
    metadata   jsonb       NOT NULL DEFAULT '{}'::jsonb,
    embedding  vector(1024) NOT NULL
);

-- Optional, but strongly recommended once the table grows:
CREATE INDEX ON learnings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON learnings (repo_name, file_path);
```

### 3. Configure

```bash
cp .env.example .env
```

Then edit `.env`. The only **required** value is `DATABASE_URL` — the app fails to start without
it. Make sure `OLLAMA_MODEL` and `EMBEDDING_DIMENSION` agree with each other *and* with the
`vector(N)` column you created above.

### 4. Run

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Run this **from the repository root**: `app` is imported from the working directory, and `.env` is
resolved relative to it as well.

Interactive docs are served by FastAPI at <http://localhost:8000/docs>. The committed contract
lives in [`openapi.json`](openapi.json) / [`openapi.yaml`](openapi.yaml).

## API

### `GET /health`

Liveness probe; reports which embedding model is configured.

```bash
curl http://localhost:8000/health
# {"status":"ok","ollama_model":"qwen3-embedding:4b"}
```

### `POST /api/v1/ingest`

Stores a file's content. Accepts `application/x-www-form-urlencoded` (or `multipart/form-data`)
with three fields: `repo_name`, `file_path`, `content`.

Existing chunks for the same `(repo_name, file_path)` pair are deleted first, so re-ingesting an
updated file is safe and idempotent. Chunks shorter than 50 characters are dropped.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
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
| `OLLAMA_URL`          | `http://localhost:11434/api/embeddings`     | embedding endpoint (local or remote)                 |
| `OLLAMA_MODEL`        | `qwen3-embedding:0.6b`                      | embedding model name                                 |
| `EMBEDDING_DIMENSION` | `384`                                       | must match the model **and** the `vector(N)` column  |
| `LANGFUSE_PUBLIC_KEY` | *(empty → tracing off)*                     | Langfuse credentials                                 |
| `LANGFUSE_SECRET_KEY` | *(empty → tracing off)*                     |                                                      |
| `LANGFUSE_HOST`       | `https://cloud.langfuse.com`                | Langfuse instance                                    |
| `APP_NAME`            | `Nescio Semantic Memory API`                | title shown in the OpenAPI docs                      |
| `LOG_LEVEL`           | `INFO`                                      | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` \| `CRITICAL` |
| `CHUNK_SIZE`          | `1000`                                      | characters per chunk                                 |
| `CHUNK_OVERLAP`       | `200`                                      | overlap between chunks — must stay **below** `CHUNK_SIZE` |
| `HOST` / `PORT`       | `localhost` / `8080`                        | *currently not applied* — pass these to `uvicorn` instead |

### Embedding models

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
uv lock                               # after changing dependencies in pyproject.toml
uv run python export_openapi.py       # regenerate openapi.json + openapi.yaml
uv run uvicorn app.main:app --reload  # dev server (from the repository root)
```

Two things are enforced by CI:

- **`uv.lock` must stay in sync with `pyproject.toml`.** If you add or change a dependency, run
  `uv lock` and commit both files.
- **The committed OpenAPI spec must match the code.** If you touch a route, a `Form(...)` field, or
  a Pydantic model, run `export_openapi.py` and commit the regenerated `openapi.json` and
  `openapi.yaml`.

Releases are automated by [release-please](https://github.com/googleapis/release-please) from
Conventional Commits (`feat:`, `fix:`, `chore:`, …). Merging the release PR it maintains bumps the
version and publishes a GitHub Release — do not edit versions by hand.

## Current limitations

This is a PoC. Known rough edges, roughly in order of how much they matter:

- **No schema management.** The `learnings` table has to be created by hand; there are no
  migrations and no bootstrap script.
- **Chunking config is not validated.** Setting `CHUNK_OVERLAP` at or above `CHUNK_SIZE` makes the
  sliding-window step zero or negative, so ingestion loops forever instead of failing fast.
- **`EMBEDDING_DIMENSION` is never checked.** It is documentation only; a mismatch with the actual
  `vector(N)` column surfaces as a database error on insert rather than at startup.
- **Langfuse keys in `.env` are ignored.** `@observe` relies on the SDK reading `LANGFUSE_*` from
  the process environment, and pydantic-settings does not export `.env` values into it. Export the
  keys in your shell or service manager, or tracing silently stays off.
- **No error handling.** A database or Ollama failure surfaces as an unhandled `500` rather than a
  meaningful HTTP error, and connections are not pooled — one is opened per request.
- **Ingestion is serial and chatty.** One blocking Ollama call per chunk, with no batching and no
  retry/backoff.
- **No authentication.** The API is meant to run on a trusted network or behind a proxy.
- **No tests.** `ci.yml` currently only verifies dependency installation and imports.

Contributions addressing any of the above are welcome.

## License

MIT — see [LICENSE](LICENSE).
