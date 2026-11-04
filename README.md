# nescio-memory

A small semantic-memory API. Feed it text, and it splits it into passages, embeds them, and stores
them in PostgreSQL + pgvector so you can later retrieve the most relevant passages by meaning as
well as by keyword.

Designed as the long-term memory layer for agents and coding assistants: `/api/v1/ingest` what a
project or document says, `/api/v1/search` it back when you need context.

> **Status: proof of concept.** The API surface, storage schema, and passage-splitting strategy are all
> expected to change significantly — see [Current limitations](#current-limitations).

## How it works

```
                              ┌──────────────────────┐
  POST /api/v1/ingest ───────▶│  passage splitter    │  note -> headed passages
                              └──────────┬───────────┘
                                         │ batches of EMBED_BATCH_SIZE
                                         ▼
                              ┌──────────────────────┐
                              │  Ollama /api/embed   │  HTTP, not in-process
                              └──────────┬───────────┘
                                         ▼
                              ┌──────────────────────┐
                              │  PostgreSQL/pgvector │  tables: notes, passages
                              │  FTS (tsv) + vectors │
                              └──────────▲───────────┘
                                         │
  POST /api/v1/search ──▶ FTS branch ────┤
                      └─▶ vector branch ─┘  fused by RRF in one statement
```

Embeddings are produced by an external [Ollama](https://ollama.com) server, so the API process
stays small and the model can run wherever you have capacity — on Kubernetes that is typically a
separate Ollama pod behind a Service. An optional in-process backend using
[sentence-transformers](https://sbert.net) is available for deployments that cannot run a second
service; see [Embedding backends](#embedding-backends). Handlers are traced with
[Langfuse](https://langfuse.com) when `LANGFUSE_*` are exported into the **process environment** —
when you run with `uv`, values that only live in `.env` are not picked up, see
[Current limitations](#current-limitations); Docker Compose exports `.env` for you.

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — manages the virtualenv and the lockfile
- **PostgreSQL 16+** with the `pgvector` extension, **`pgvector` >= 0.8.0 required** — search sets
  `hnsw.iterative_scan`, which earlier versions reject outright
- **Docker** — to run the test suite, which starts a real PostgreSQL + pgvector container, and
  for the optional [Docker Compose](#5-run-with-docker-compose) setup
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

This creates the `vector` extension, the `notes`, `passages` and `api_keys` tables, and every
index, including the GIN index behind keyword search and the HNSW index behind vector search. Do not create tables by hand — the models and
the migrations are checked against each other, and a hand-built schema will be missing columns the
application requires.

The application also verifies at startup that `EMBEDDING_DIMENSION` matches the
`passages.embedding` column, so it will not start against an unmigrated or
mismatched database. That means a reachable database is required to boot.

Upgrading from 0.5.x? Read [Upgrading to 0.6.0](#upgrading-to-060) first — the migration drops the
v1 storage.

### 4. Create an API key

Every `/api/v1` route requires an API key. Mint one per client:

```bash
uv run python -m scripts.create_api_key my-client
```

```
client_name : my-client
api_key     : nm_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
⚠️  Store this now — it cannot be retrieved again.
```

Only a SHA-256 hash of the key is stored, so the plaintext cannot be recovered — mint a new key if
you lose it. Omit the argument and the client is named `default`.

#### Key rotation

Rotate a client's key without SQL, in three steps:

```bash
# 1. Mint the replacement. Copy its api_key line now: it is shown once.
uv run python -m scripts.manage_api_keys create my-client

# 2. Deploy the new key to the client and check that it authenticates.

# 3. Find the old key's id, then revoke it.
uv run python -m scripts.manage_api_keys list --client my-client
uv run python -m scripts.manage_api_keys revoke --id 1
```

`list` shows each key's id, client, display prefix (the first 15 characters, so you can match a row to
the key you copied), created time and revoked time. It never shows the secret or its hash. Revoked keys
are hidden unless you pass `--include-revoked`.

- `revoke --id N` revokes one key. Revoking a key that is already revoked prints `revoked 0` and exits 0.
- `revoke --client NAME --yes` revokes every active key of that client. It refuses without `--yes`.

A revoked key is rejected with `401` on its next request. Revoking sets `revoked_at` rather than
deleting the row, so the audit trail is kept.

The `client_name` is not just a label: every ingested note and passage is tagged with it, and each client sees
only its own data. See [Authentication](#authentication).

### 5. Run with Docker Compose

The [`Dockerfile`](Dockerfile) and [`compose.yaml`](compose.yaml) run PostgreSQL + pgvector,
Ollama, the schema migration and the API as one stack, so with containers you can skip steps 1
and 3 (the `migrate` service applies the schema). To run the API from a Python checkout instead,
see [Development](#development).

```bash
docker compose up -d
docker compose exec app python scripts/create_api_key.py my-client   # step 4, inside the container
curl http://localhost:8080/ready
```

`.env` is optional here: every value has a default. Copy `.env.example` to `.env` to change
`OLLAMA_MODEL`, enable Langfuse and so on.

The API is published on port **8080** (`APP_PORT` changes the host side); interactive docs are at
<http://localhost:8080/docs>. The first `up` pulls the Ollama image and downloads the embedding
model, so `/ready` reports `degraded` until that finishes. The committed contract lives in
[`openapi.json`](openapi.json) / [`openapi.yaml`](openapi.yaml).

| Service       | What it does                                                                                 |
| ------------- | -------------------------------------------------------------------------------------------- |
| `db`          | `pgvector/pgvector` (PostgreSQL 17, pgvector 0.8.x), data in the `pgdata` volume. Not published to the host. |
| `ollama`      | Ollama server, models in the `ollama` volume. For a GPU, uncomment the `deploy` block in `compose.yaml`. |
| `ollama-pull` | One-shot: downloads `OLLAMA_MODEL` into the volume, then exits.                              |
| `migrate`     | One-shot: runs `alembic upgrade head` once `db` is healthy, then exits.                      |
| `app`         | The API. Starts only after `migrate` exited successfully.                                    |

- **Migrations never run on app start.** Only the `migrate` service applies them. After upgrading
  the image, run `docker compose run --rm migrate` (or just `docker compose up -d`, which re-runs
  it) before the new `app` starts.
- **The `app` health check calls `/ready`**, so the container is healthy only when PostgreSQL
  answers. A missing Ollama gives `degraded` (still healthy); a dead database makes it unhealthy.
  The image's own `HEALTHCHECK` calls `/health`, which touches no dependency.
- **Ollama is optional for startup.** `docker compose up -d db migrate app` runs the API without it;
  search then falls back to keyword-only and ingest fails until an embedding backend is reachable.
- **`.env` is passed into the container whole**, including `LANGFUSE_*`. Compose exports it into
  the process environment, which is where the Langfuse SDK reads it, so the
  [`.env` limitation](#current-limitations) does not apply here. `DATABASE_URL` and `OLLAMA_URL`
  are overridden to point at the `db` and `ollama` services, whatever `.env` says. Database
  credentials come from `POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_DB` (default `nescio`,
  `nescio`, `nescio_memory`); set a real password if the stack is reachable from anywhere but
  your own machine, and keep it URL-safe because it is placed into `DATABASE_URL`.
- **Image variants.** The default image uses Ollama and has no torch. For the in-process backend,
  build with the `local-embeddings` extra:

  ```bash
  EXTRAS=local-embeddings docker compose build
  ```

  and set `EMBEDDING_BACKEND=local`; read [Embedding backends](#embedding-backends) first, since
  that backend has a different embedding width. Only the default image is published.
- **Published image.** Each GitHub release pushes `ghcr.io/noctua84/nescio-memory` tagged `X.Y.Z`,
  `X.Y` and `latest`. Pull requests build the image and smoke-test the stack in CI
  ([`docker.yml`](.github/workflows/docker.yml)).

The container runs as an unprivileged user (uid 10001) and listens on port 8080 inside, regardless
of `HOST` and `PORT` in `.env` (see [Configuration](#configuration)).

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

### `GET /ready`

Readiness probe. Where `/health` asks "is the process alive?" and contacts nothing, `/ready` asks
"can it serve right now?" and checks PostgreSQL (`SELECT 1` under a statement timeout) and the
embedding backend (Ollama `POST /api/show` for `OLLAMA_MODEL`, which proves the server is up and
the model is pulled without loading it; the `local` backend runs in-process and reports
`not_checked`). Use `/health` for restart decisions and `/ready` for traffic and monitoring: a
dead dependency should take a replica out of rotation, not get it killed.

```bash
curl -i http://localhost:8000/ready
# HTTP/1.1 200 OK
# {"status":"ready","checks":{"database":{"status":"ok"},
#  "embedding_backend":{"status":"ok"}}}
```

| Overall `status` | HTTP | Meaning                                                               |
| ---------------- | ---- | --------------------------------------------------------------------- |
| `ready`          | 200  | database `ok`; embedding backend `ok` or `not_checked`                 |
| `degraded`       | 200  | database `ok`, embedding backend `fail` or `timeout`                   |
| `not_ready`      | 503  | database `fail` or `timeout`                                           |

Each check reports `ok`, `fail` or `timeout` (`not_checked` for the `local` embedding backend).
Neither the exception text nor any host name is returned; the detail is logged at `WARNING`.

An Ollama outage is deliberately `200 degraded`, not `503`: Ollama is shared by every replica, so
failing readiness on it would eject all of them at once, and search already degrades to keyword
results without it. The consequence is that an orchestrator only ever sees the `200`, so
**alert on `status` not being `ready`**, and read `checks.embedding_backend.status` to see which
dependency failed.

The probe is safe to poll hard and is unauthenticated (no `X-API-Key`):

- The response arrives within `max(READY_DB_TIMEOUT_S, READY_EMBEDDING_TIMEOUT_S)` plus a small
  overhead, even if a dependency hangs; a check that has not answered by then reports `timeout`.
- At most one check per dependency is ever in flight. Concurrent requests share it, and a hung
  dependency keeps one background thread busy until its own connect or HTTP timeout fires instead
  of piling up connections.
- A finished result is reused for `READY_CACHE_TTL_S`, so a readiness failure can take up to that
  long to be reported and to clear.

### Authentication

Every route under `/api/v1` requires an `X-API-Key` header. `GET /health` and `GET /ready` do not,
so a liveness or readiness probe needs no credentials.

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: nm_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{"q": "...", "top_k": 3}'
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

The key does more than authenticate. Every note and passage written through it is tagged with its
`client_name`, and that tag scopes every read and delete:

- Search returns only the calling client's passages.
- `GET` and `DELETE` on a note id treat another client's notes as if they do not exist (the same
  `404`).
- Re-ingesting a note replaces only the calling client's copy of it.
- Two clients can hold the same `(scope, path_or_uri)` without seeing or overwriting each
  other's note.

### Notes and passages

Design rationale and the retrieval-quality baseline live in
[`docs/superpowers/specs/2026-10-29-notes-passages-slice-1-design.md`](docs/superpowers/specs/2026-10-29-notes-passages-slice-1-design.md).
The routes are tagged `Ingest`, `Notes` and `Search` in the OpenAPI document, with operation ids
`ingestNote`, `getNote`, `deleteNote`, `deleteNoteByPath`, `deleteScope` and `searchPassages`.

Every `/api/v1` route requires the `X-API-Key` header (only `GET /health` and `GET /ready`, which
sit outside `/api/v1`, do not), and all data is tenant-scoped by it.

#### `POST /api/v1/ingest`

Stores one note as JSON (not `Form(...)` — there is no 1 MiB per-field byte limit here, only
`MAX_CONTENT_CHARS`, 500,000 characters, otherwise `400`). The note is split into passages,
embedded in batches, and both are written in one transaction.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-API-Key: nm_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "scope": "nescio-memory",
    "path_or_uri": "docs/architecture.md",
    "content": "# Architecture\n\n...",
    "status": "active",
    "meta": {}
  }'
```

```json
{
  "note_id": 42,
  "passages": 3,
  "skipped": false
}
```

| Field         | Description                                                                 |
| ------------- | ---------------------------------------------------------------------------- |
| `scope`       | also accepted as `repo_name`; defaults to `""`                               |
| `path_or_uri` | a relative path (no leading `/`, no `..`) or an absolute URI                 |
| `content`     | note body, UTF-8 text                                                        |
| `status`      | optional, stored as given — the service does not derive it from frontmatter |
| `meta`        | optional, stored as given                                                    |
| `force`       | optional, default `false` — bypasses the skip below without changing any stored value |

Re-ingesting the same `(scope, path_or_uri)` is **skipped** — `note_id` and `passages` are
returned unchanged, `skipped: true`, and nothing is re-split or re-embedded — only when **both**
the content hash **and** the pipeline version (which covers the splitter, the FTS composition,
the embedding-input composition, and the embedding backend + dimension) match the stored note. A
hash match behind a changed pipeline version still re-splits and re-embeds. `force: true` is the
escape hatch for a change the pipeline version cannot see — most plausibly a re-pulled embedding
model behind an unchanged backend name and dimension — and always costs the work, never skipping.

#### `GET /api/v1/notes/{id}`

Returns one note by id, scoped to the calling client — a note belonging to another client is
**indistinguishable** from an id that does not exist at all: both are the same `404`.

```bash
curl http://localhost:8000/api/v1/notes/42 -H "X-API-Key: nm_your_key_here"
```

```json
{
  "id": 42,
  "scope": "nescio-memory",
  "path_or_uri": "docs/architecture.md",
  "name": "Architecture",
  "description": null,
  "type": null,
  "provenance_kind": null,
  "provenance_date": null,
  "status": "active",
  "content_hash": "...",
  "pipeline_version": "...",
  "body": "# Architecture\n\n...",
  "meta": {},
  "passage_count": 3,
  "created_at": "2026-10-29T00:00:00Z",
  "updated_at": "2026-10-29T00:00:00Z"
}
```

`body` already contains the text of every passage, so passage texts are not returned alongside it
— `passage_count` is the count, computed under the same tenant filter.

#### `DELETE /api/v1/notes/{id}`

The erasure path: removes a note and, by cascade, every one of its passages. Returns `204` with
no body on success, and the same indistinguishable `404` as `GET` for a missing or foreign id.
Nothing reconciles notes against a source tree, so a client that stops holding a file must call a
delete route itself — this one if it kept the id, or the two below if it only has the path or the
scope.

```bash
curl -X DELETE http://localhost:8000/api/v1/notes/42 -H "X-API-Key: nm_your_key_here"
```

#### `DELETE /api/v1/notes?scope=…&path_or_uri=…`

Deletes the one note stored at `(scope, path_or_uri)`, for a client that does not hold the id. Both
query parameters are required; `scope` may be the empty string, which is the root scope. They are
query parameters rather than path segments because a scope can be empty and a path contains
slashes. `path_or_uri` follows the same rule as ingest (a relative path without `..`, or a
`scheme://` URI), so a value ingest would refuse is refused here too.

```bash
curl -X DELETE -G http://localhost:8000/api/v1/notes -H "X-API-Key: nm_your_key_here" \
  --data-urlencode "scope=nescio-memory" \
  --data-urlencode "path_or_uri=docs/architecture.md"
```

```json
{
  "notes_deleted": 1,
  "passages_deleted": 3,
  "notes": [
    {"note_id": 42, "path_or_uri": "docs/architecture.md", "content_hash": "..."}
  ]
}
```

No match is `404 {"detail": "Note not found"}`, byte-identical to delete by id and the same whether
the note never existed or belongs to another client. The lookup is scoped to the calling client, so
it cannot be used to probe another client's paths. Unlike delete by id the success response is
`200` with a body, so the caller learns what was removed (the id and content hash are gone from the
database by then).

#### `DELETE /api/v1/scopes?scope=…`

Retires a whole scope: deletes every note in it and, by cascade, their passages. The response has
the same shape as delete by path, with one entry in `notes` per removed note. A scope with no
notes — including one that belongs to another client — is **not** an error: the answer is `200`
with `notes_deleted` and `passages_deleted` of `0` and an empty `notes` list, which keeps a retry
after a partial failure harmless.

```bash
curl -X DELETE "http://localhost:8000/api/v1/scopes?scope=nescio-memory" -H "X-API-Key: nm_your_key_here"
```

**The root scope cannot be deleted by accident.** Notes ingested without a scope live in the empty
scope `""`, and a client that builds this URL from a variable that turned out empty would
otherwise erase them. So an empty or omitted `scope` is refused with `400` before anything runs.
To delete the root scope on purpose, say so:

```bash
curl -X DELETE "http://localhost:8000/api/v1/scopes?root_scope=true" -H "X-API-Key: nm_your_key_here"
```

With `root_scope=true`, `scope` must be empty or omitted and only the root scope is deleted;
combining `root_scope=true` with a non-empty `scope` is a `400` as well.

#### `POST /api/v1/search`

Hybrid search: a full-text branch and a vector branch are fused by reciprocal rank (RRF) in a
single SQL statement. Results are **passages**, each carrying its note's identity, description and
provenance.

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: nm_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{"q": "invoice-first checkout", "top_k": 5}'
```

```json
{
  "results": [
    {
      "passage_id": 101,
      "note_id": 42,
      "ordinal": 0,
      "name": "Architecture",
      "description": null,
      "type": null,
      "path_or_uri": "docs/architecture.md",
      "heading_path": "Architecture › Overview",
      "content": "...",
      "provenance_kind": null,
      "provenance_date": null,
      "status": "active",
      "score": 0.0325,
      "fts_rank": 1,
      "vector_rank": 3
    }
  ],
  "degraded": false
}
```

| Field         | Default | Description                                                           |
| ------------- | ------- | ----------------------------------------------------------------------- |
| `q`           | —       | also accepted as `query` (required); parsed with `websearch_to_tsquery` semantics for the keyword branch — quoted phrases, `OR`, leading `-` to exclude a word — and embedded for the vector branch |
| `top_k`       | `5`     | maximum number of results, 1–50, otherwise `422`                        |
| `repo_filter` | `null`  | restrict to notes with that `scope`; `null` or an empty string `""` both mean no filter (all scopes). Notes ingested without a scope have scope `""` and cannot be selected on their own |

How a result is scored:

- Each branch produces up to `SEARCH_CANDIDATES` candidates (at least `top_k`), already filtered
  by client and, when given, by scope — before that branch's own `LIMIT`, so another tenant's rows
  can never crowd out yours.
- A passage's `score` is the sum, over the branches that found it, of
  `weight / (RRF_K + rank)`, with `FTS_WEIGHT` and `VECTOR_WEIGHT` as the weights. It is
  comparable only within one response; higher is more relevant.
- `fts_rank` and `vector_rank` are the 1-based position within each branch, or `null` when the
  passage was not among that branch's candidates or the branch did not run. A `null` `fts_rank` is a
  semantic-only match.
- There is **no relevance floor**: the vector branch always returns its nearest neighbours, so
  results fill up to `top_k` whenever that many passages exist and trailing ones can be weak.
- `degraded` is always present. It is `true` when the query could not be embedded because the
  embedding backend was transiently unavailable (bounded by `QUERY_EMBED_TIMEOUT_S`; HTTP 408, 429,
  500, 502, 503, 504, a timeout or a connection failure); the results are then keyword-only and
  every `vector_rank` is `null`. Any other embedding failure still returns an error.

The keyword branch ranks by `ts_rank_cd` over three FTS weights: `A` (note name + heading path), `B`
(passage content) and `C` (note description). `description` is deliberately weighted below
`content`: it is note-level and duplicated onto every passage of the note, so at the same weight
as `content` it would score every passage of a long note identically and could outrank a passage
that genuinely discusses the term — the "returns a page to read again" failure this search exists
to avoid. A query that reduces to an empty tsquery (a stop word alone, or only punctuation) is not
an error: the keyword branch contributes nothing and the vector branch still answers it, so the
response is a `200` with semantic matches (and an empty `results` only if the caller holds no
passages).

### Error responses

Errors from `/api/v1` carry a JSON body, `{"detail": ...}`. Except for `422`, `detail` is a string: a
fixed message for each failure below, never an exception message.

| Status | `detail` | Retry-After (seconds) | When |
| ------ | -------- | --------------------- | ---- |
| 400 | `Request contains data the database cannot accept` | none | The database rejected a value, such as a NUL byte in text. |
| 400 | `path_or_uri must be a relative path without '..', or an absolute URI` | none | `POST /api/v1/ingest` with an absolute path or a `..` component. |
| 400 | `content is too large: N characters exceeds the 500000 limit.` | none | `POST /api/v1/ingest` with `content` over 500,000 characters. `N` is the submitted length. |
| 400 | `Invalid Content-Length header.` | none | Any `/api/v1` route, with a `Content-Length` that is not a non-negative integer. Rejected before authentication. |
| 401 | `Missing API key` | none | The `X-API-Key` header is absent. |
| 401 | `Invalid or revoked API key` | none | The key is unknown or revoked. |
| 404 | `Note not found` | none | `GET` or `DELETE /api/v1/notes/{id}` for a note that does not exist, or belongs to another client. |
| 413 | `Request body too large: limit is 8388608 bytes.` | none | Any `/api/v1` route, with a body over the limit. Rejected before authentication. |
| 422 | list of error objects | none | Request validation failed. Each object has `loc`, `msg` and `type`. |
| 500 | `Embedding backend is misconfigured` | none | The embedding backend cannot work as deployed. Retrying will not help; the deployment must change. |
| 500 | `Internal server error` | none | An unanticipated failure. The traceback is in the server log. |
| 503 | `Database unavailable` | 30 | The database cannot be reached. |
| 503 | `Query exceeded time limit` | none | A query was cancelled, in practice by the search `statement_timeout` (`STATEMENT_TIMEOUT_MS`). |
| 503 | `Embedding backend unavailable` | 30 | The embedding backend cannot be reached, timed out, or answered with a transient HTTP status. |
| 503 | `Embedding backend returned an unexpected response` | 30 | The embedding backend answered with something that is not a usable vector. |

Retry-After is sent only when a retry can succeed. A dependency that is down may recover, so the 30
seconds is a hint to back off, not a promise. A bad request, a misconfiguration and a bug fail the
same way on every retry, so their responses carry no header.

A transient embedding failure is handled differently by the two routes that embed. `POST /api/v1/ingest`
returns the `503` above. `POST /api/v1/search` answers `200` with full-text results and
`degraded: true` instead (see `degraded` under `POST /api/v1/search`). A bad response or a
misconfiguration is not degraded on either route.

## Upgrading to 0.6.0

0.6.0 is a breaking release.

- **The v1 learnings API is gone.** The multipart `POST /api/v1/ingest`, the `/api/v1/search`
  with context expansion (`similarity`, `context_ref`, `contexts`) and the `learnings` table
  (dropped by migration `3d67961c1596`) no longer exist. The data is not migrated — the old rows
  were overlapping fixed-size fragments and cannot be turned into notes. Re-ingest from source with
  [`scripts/reingest.py`](#re-ingesting-a-corpus).
- **The passages API moved from `/api/v2/*` to `/api/v1/*`.** The contract is the former v2
  one, not the old v1: a JSON body, `q` (not `query`), `scope` (not `repo_name`), and the `rank`
  field replaced by `score`, `fts_rank` and `vector_rank` plus a top-level `degraded`. OpenAPI
  schema names lost their `V2` suffix, and the operation ids are `ingestNote`, `getNote`,
  `deleteNote` and `searchPassages`.
- **Removed settings:** `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_CONTEXT_CHARS` and
  `MAX_CONTEXT_DOCUMENTS`. Unknown keys are ignored, so an old `.env` still boots.
- **`OLLAMA_URL` must end in `/api/embed`.** Ingest embeds in batches through that endpoint; the
  legacy `/api/embeddings` and a bare host with no path are rejected at startup. Change the path and keep the same host.
- **New settings** for hybrid search and batching are listed under
  [Configuration](#configuration).

## Configuration

All settings are read from `.env` via [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/);
see [`.env.example`](.env.example) for the annotated template. Unknown keys are ignored.

| Variable              | Default                                    | Description                                          |
| --------------------- | ------------------------------------------ | ---------------------------------------------------- |
| `DATABASE_URL`        | *required*                                 | PostgreSQL connection string; a plain `postgresql://` uses the bundled psycopg 3 driver |
| `DB_CONNECT_TIMEOUT_S` | `10`                                       | seconds per connection attempt (applies per resolved address); with `pool_pre_ping` a dead database adds this inside a request; must be `>= 2`, otherwise the app refuses to start |
| `STATEMENT_TIMEOUT_MS` | `5000`                                     | caps each statement, not the whole request; must be `1..60000`; a hybrid search is one statement under this cap; **breaking in 0.5.0**: values above 60000 now refuse to start |
| `EMBEDDING_BACKEND`   | `ollama`                                    | `ollama` (HTTP) or `local` (in-process); any other value fails at startup |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2`  | model for the `local` backend only — see [Embedding backends](#embedding-backends) |
| `OLLAMA_URL`          | `http://localhost:11434/api/embed`          | Ollama batch embedding endpoint (local or remote); must end in `/api/embed`, with no trailing slash (a reverse-proxy prefix such as `/ollama/api/embed` is fine) — a bare host (`http://host:11434`), any other path, the legacy `/api/embeddings` and a trailing slash (`/api/embed/`) fail at startup |
| `OLLAMA_MODEL`        | `qwen3-embedding:0.6b`                      | embedding model name                                 |
| `EMBEDDING_DIMENSION` | `1024`                                      | must match the model **and** the `vector(N)` column — the default is the width `qwen3-embedding:0.6b` actually emits; verify with [`scripts/check_embedding_dimension.py`](scripts/check_embedding_dimension.py) |
| `LANGFUSE_PUBLIC_KEY` | *(empty → tracing off)*                     | Langfuse credentials                                 |
| `LANGFUSE_SECRET_KEY` | *(empty → tracing off)*                     |                                                      |
| `LANGFUSE_HOST`       | `https://cloud.langfuse.com`                | Langfuse instance                                    |
| `APP_NAME`            | `Nescio Semantic Memory API`                | title shown in the OpenAPI docs                      |
| `LOG_LEVEL`           | `INFO`                                      | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` \| `CRITICAL` (case-insensitive; anything else fails at startup); applies to the `app.*` and `uvicorn.*` loggers, the root logger stays at `WARNING` so httpx and SQLAlchemy stay quiet — see [Logging](#logging) |
| `LOG_FORMAT`          | `json`                                      | `json` (one OpenTelemetry-shaped object per line, for a collector) or `text` (human-readable, for a local terminal); any other value fails at startup |
| `OTEL_SERVICE_NAME`   | *(unset → `APP_NAME`)*                      | `service.name` on every log line. **Process environment only**, not read from `.env`, like `LANGFUSE_*`: Langfuse's tracer reads it from the process environment too, so logs and traces carry one service identity |
| `OTEL_RESOURCE_ATTRIBUTES` | *(unset)*                              | extra resource fields on every log line, as comma-separated `key=value` pairs (percent-decoded); `service.name` and `service.version` inside it are honoured. A malformed entry is skipped with one warning. **Process environment only**, not read from `.env` |
| `SEARCH_CANDIDATES`   | `50`                                        | candidates each search branch (keyword, vector) contributes before fusion; `1..1000`; the effective number is `max(SEARCH_CANDIDATES, top_k)` |
| `RRF_K`               | `60`                                        | reciprocal-rank-fusion damping constant, `>= 1`; a passage scores `weight / (RRF_K + rank)` per branch |
| `FTS_WEIGHT` / `VECTOR_WEIGHT` | `1.0` / `1.0`                      | per-branch multipliers on the fused score, finite and `>= 0`; `0` switches that branch off (`VECTOR_WEIGHT=0` is the keyword-only baseline); both `0` is rejected at startup |
| `QUERY_INSTRUCTION`   | *(Qwen3 retrieval instruction)*             | prefix added to **query** embeddings only, never to passages, for instruction-tuned models such as Qwen3-Embedding; empty string disables it. Set `QUERY_INSTRUCTION=""` for a model that is not instruction-tuned (the `local` backend's sentence-transformers model, or any non-Qwen3 Ollama model): the prefix is applied to queries only and would skew ranking |
| `QUERY_EMBED_TIMEOUT_S` | `15`                                      | seconds a search waits to embed the query, `0 < x <= 30`; on a transient failure the search degrades to keyword-only (`degraded: true`). The default covers a cold model load: Ollama unloads an idle model after its `keep_alive` (5 minutes by default), so the first search after an idle period waits for a reload. A startup warm-up would not help, since it only covers the first load after a deploy; to avoid the reload, set `OLLAMA_KEEP_ALIVE` on the Ollama server. A longer wait does not hold a database connection |
| `EMBED_BATCH_SIZE`    | `16`                                        | passages per `/api/embed` call at ingest, `1..256` |
| `READY_DB_TIMEOUT_S`  | `2.0`                                       | seconds `GET /ready` waits for the database check before reporting `timeout`, `0 < x <= 10`; see [`GET /ready`](#get-ready) |
| `READY_EMBEDDING_TIMEOUT_S` | `2.0`                                 | seconds `GET /ready` waits for the Ollama check before reporting `timeout`, `0 < x <= 10` |
| `READY_CACHE_TTL_S`   | `5.0`                                       | seconds a finished readiness result is reused, `0 <= x <= 60`; `0` disables reuse (a check already running is still shared) |
| `EMBED_RETRY_ATTEMPTS` | `3`                                        | tries per embedding call, the first included, `1..5`; `1` means no retry. Only transient failures are retried (timeouts, connection errors, HTTP 408/429/500/502/503/504); a bad response or a misconfiguration never is |
| `EMBED_RETRY_BASE_DELAY_S` | `0.5`                                  | backoff before the first retry, `0 < x <= EMBED_RETRY_MAX_DELAY_S`; it doubles each retry and the actual wait is uniformly random below that (full jitter) |
| `EMBED_RETRY_MAX_DELAY_S` | `4.0`                                   | longest single backoff, `0 < x <= 10`; also the cap on a `Retry-After` the embedding server sends with a 429 or 503 |
| `EMBED_RETRY_BUDGET`  | `2`                                         | retries shared by **all** calls of one ingest request, `0..10`; `0` disables retry. See "Ingestion retries" under Current limitations for the worst-case bound |
| `HOST` / `PORT`       | `localhost` / `8080`                        | *currently not applied* — pass these to `uvicorn` instead |

### Logging

Every record goes to **stdout** as one line of JSON (`LOG_FORMAT=json`, the default): from the
`app.*` loggers and from `uvicorn.error` and `uvicorn.access`. A traceback is one escaped string
inside the line, never several lines. Fields:

| Field | Value |
| ----- | ----- |
| `timestamp` | ISO 8601, UTC, milliseconds, `Z` suffix |
| `level`, `severity_number` | `DEBUG` 5, `INFO` 9, `WARNING` 13, `ERROR` 17, `CRITICAL` 21 (OpenTelemetry severity numbers; a level in between maps to the nearest lower one) |
| `logger`, `message` | the logger name and the formatted message |
| `service.name`, `service.version` | see `OTEL_SERVICE_NAME` and `OTEL_RESOURCE_ATTRIBUTES` above; the version is the project version |
| `trace_id`, `span_id` | 32 and 16 hex characters, present only while an OpenTelemetry span is active |
| `exception.type`, `exception.message`, `exception.stacktrace` | present when an exception is logged |
| anything passed as `extra=` | emitted as its own field |

Request logs from `uvicorn.access` carry `http.request.method`, `url.path`, `http.response.status_code`,
`client.address` and `client.port`, plus a message rebuilt from them. **The query string is dropped**:
it can carry a note path or a search term, so the raw uvicorn message is never emitted. Request
bodies, note content, paths and API keys are not logged, and the keys `content`, `body`, `q`,
`query`, `api_key`, `x_api_key`, `headers` and `path_or_uri` are replaced with `[redacted]` if a call
site ever passes them.

**Delivery is stdout only.** To get the records into Dash0 (or any OpenTelemetry backend), point a
collector's `filelog` receiver at the container's stdout. There is deliberately no OTLP log
exporter: exporting as well would deliver every record twice.

**Trace correlation.** `trace_id` and `span_id` come from `opentelemetry.trace.get_current_span()`.
Inside a route decorated with Langfuse's `@observe`, that is the Langfuse span (checked against
langfuse 5.0.4: a valid, recording span with real ids) **when the `LANGFUSE_*` keys are set**. With
the keys unset, or with `LANGFUSE_TRACING_ENABLED=false`, Langfuse hands out a non-recording span
with an invalid (all-zero) context, so the log lines carry no `trace_id`/`span_id`. That is correct,
since there is no trace to link to, but it means correlation needs Langfuse tracing switched on.

### Embedding backends

|                  | `ollama` (default)                    | `local`                                        |
| ---------------- | ------------------------------------- | ---------------------------------------------- |
| Runs             | in a separate Ollama process or pod   | in-process, via sentence-transformers          |
| Install          | nothing extra                         | `uv sync --extra local-embeddings` (adds torch) |
| Model weights    | managed by Ollama                     | fetched from Hugging Face Hub on first use     |
| Scales           | independently of the API              | with the API process — needs CPU/GPU and RAM there |
| Best for         | clusters that already run Ollama      | single-process or offline deployments          |

Select the backend with `EMBEDDING_BACKEND`. The `local` backend loads its model once and caches
it for the lifetime of the process. The default `QUERY_INSTRUCTION` targets Qwen3-Embedding, so
with the `local` backend or any other non-instruction-tuned model, set `QUERY_INSTRUCTION=""`.
Selecting `local` without installing the extra fails on the
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
| `local`                   | `all-MiniLM-L6-v2`            | **384**  | none shipped — see below                         |

`passages.embedding` is `vector(1024)` in every shipped revision, and the app checks it against
`EMBEDDING_DIMENSION` at startup. Running the `local` backend with its default 384-wide model
therefore needs a migration of your own that changes the column width (see the generated-column
note below), `EMBEDDING_DIMENSION=384`, and a re-ingest. Do not stop `alembic upgrade` at an older
revision to get a 384-wide schema: the last 384-wide revision (`3d80eb22a17d`) predates the
`notes` and `passages` tables, so it has nothing for the API to write to. With the stock
`EMBEDDING_DIMENSION=1024` the `local` backend fails every embedding call, loudly, on the dimension
check — the mirror image of the defect in #36.

Pointing `LOCAL_EMBEDDING_MODEL` at a genuinely 1024-dimensional sentence-transformers model
would make one schema serve both, and is the obvious next step; it is not taken here because
nobody has run it. An unexecuted dimension claim in this file is what caused #36 in the first
place.

#### Changing the embedding width is blocked by the stored generated column

`passages.embedding_degenerate` is a `STORED GENERATED` column computed as
`(embedding <=> embedding) = 'NaN'::float8` (added in `17be352b1a4c`). It flags stored vectors
whose self-distance is NaN (zero-norm vectors), so the vector arm of search excludes them without
recomputing the test per query; such a passage can still be found through the full-text arm.
Postgres refuses `ALTER COLUMN embedding TYPE vector(N)` while any generated column reads
`embedding` — the column's expression is bound to the old type. A width-change migration must, in
order: drop `embedding_degenerate`, run the `ALTER COLUMN ... TYPE vector(N)`, then re-add
`embedding_degenerate` with the same expression (`GENERATED ALWAYS AS ((embedding <=> embedding)
= 'NaN'::float8) STORED`).

### Ollama models

| Model                   | Dimensions | Rough cost          |
| ----------------------- | ---------- | ------------------- |
| `qwen3-embedding:0.6b`  | 1024       | ~400 MB RAM         |
| `qwen3-embedding:4b`    | 2560       | ~2.5 GB RAM         |
| `qwen3-embedding:8b`    | 4096       | GPU recommended     |

Widths taken from the models' own `hidden_size` (`Qwen/Qwen3-Embedding-{0.6B,4B,8B}`), and 0.6b
confirmed against a running server: `/api/show` reports `qwen3.embedding_length: 1024` and
`/api/embed` returns 1024-float vectors. The previous version of this table said 384 and 1024 for
the first two rows, which is where #36 came from — do not re-derive a width from this file
without running `python -m scripts.check_embedding_dimension` against the model.

Switching models means recreating the `embedding` column at the new dimension and re-ingesting
everything — vectors of different dimensions are not comparable.

### Checking the dimension

`EMBEDDING_DIMENSION` has to agree with two independent things, and until #36 only one of them
was ever checked:

| Must agree with                       | Checked by                                                  | When            |
| ------------------------------------- | ----------------------------------------------------------- | --------------- |
| the `passages.embedding` column       | `verify_embedding_dimension` (`app/core/schema_checks.py`)   | every startup   |
| the SQLAlchemy model                  | `tests/test_schema_drift.py`                                | every CI run    |
| **what the model actually emits**     | `scripts/check_embedding_dimension.py`                      | **on demand**   |

The third row is the gap that made the stock configuration inert while the suite stayed green: the
tests embed through a fake that returns `EMBEDDING_DIMENSION` floats by construction, so they agree
with the setting no matter what it says. Run the script after changing `OLLAMA_MODEL`,
`LOCAL_EMBEDDING_MODEL`, `EMBEDDING_BACKEND` or `EMBEDDING_DIMENSION`:

```bash
uv run python -m scripts.check_embedding_dimension
# OK  ollama backend at http://localhost:11434/api/embed, model qwen3-embedding:0.6b
#   emits 1024 dimensions, matching EMBEDDING_DIMENSION
```

It exits `0` on a match, `1` on a mismatch and `2` if the backend could not be reached, so a
deployment pipeline can treat "wrong" and "could not tell" differently.

This is a script and not a startup check on purpose: probing at boot would make the API's liveness
depend on the embedding backend, and an API that will not start because Ollama is restarting is a
worse failure than the one being guarded against. `tests/test_embedding_width_contract.py` runs the
same probe in the suite and **skips** when no backend is reachable — the skip is printed in every
run's summary (`addopts = "-ra"`), because a silent skip would leave the same hole.

## Backup and recovery

Every night a systemd timer runs three steps, each only if the one before succeeded:

1. **Back up** ([`scripts/backup.py`](scripts/backup.py)): `pg_dump -Fc` into the spool, a sidecar
   manifest (sha256, alembic revision, per-client note and passage counts, pgvector version), upload
   of both to an rclone **crypt** remote, then pruning of the remote and the spool.
2. **Restore check** ([`scripts/restore_check.py`](scripts/restore_check.py)): fetch the newest dump
   back from the remote, check its sha256, restore it into a scratch database, and verify it. The
   alembic revision and every client's counts must match the manifest, the HNSW index must exist and
   be valid, and a stored passage's embedding used as the query must return that passage's note as
   the top hit, both through the production search and straight from the index with the planner
   forced onto it. This needs no Ollama. The scratch database is then dropped.
3. **Heartbeat**: a `GET` of `BACKUP_HEARTBEAT_URL`, only if all of that passed. A dead-man's switch
   such as [healthchecks.io](https://healthchecks.io) alerts when the ping stops arriving, so **silence
   is the alert**: a broken timer, a dead VPS and a corrupt dump look the same and all of them page you.

Set the switch's period to 1 day and its grace to more than a whole run takes: the timer starts up to
15 minutes late, and the run's time is in the journal as `elapsed_s` plus the dump and the upload.

The manifest's counts and the dump come from one exported snapshot, so an ingest during the backup
cannot make a good backup fail its check. The check refuses a newest dump with no manifest rather than
falling back to an older backup, which would keep the heartbeat alive while nothing new is arriving. A
database with no passages also fails: it restores perfectly and proves nothing about the embeddings.

**Retention:** the newest backup of each of the `BACKUP_RETENTION_DAILY` most recent days that have
one, and of each of the `BACKUP_RETENTION_WEEKLY` most recent ISO weeks that have one (the windows
overlap). It counts backups, not the calendar, so when backups stop the last good ones do not age out.
Only file names this tool writes are ever deleted. The spool is pruned before the upload, so failing
uploads cannot fill the disk.

**What the logs hold:** the step, its exit code, counts and timings. A tool's stderr is never printed,
because `pg_restore` quotes the failing `COPY` line, which is note content, and rclone quotes paths.
When you need the cause, repeat the step by hand with the commands below.

### Backup configuration

`BackupSettings` in [`scripts/backup_settings.py`](scripts/backup_settings.py), read from the
environment or `.env`; separate from the API's [Configuration](#configuration), which does not read
these. Invalid values stop the script before it touches anything, with exit code `2`.

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | *required* | the source, the same variable the API uses |
| `RESTORE_CHECK_DATABASE_URL` | *required* | a maintenance database (`postgres`) on the server that hosts the scratch database. The role must be a **superuser**: the restore runs `CREATE EXTENSION vector` and pgvector is not a trusted extension. The server needs pgvector at the source's version or newer; the check says so if not |
| `BACKUP_RCLONE_REMOTE` | *required* | `name:path` of an rclone **crypt** remote. The dumps hold personal notes, so any other remote type is refused |
| `BACKUP_HEARTBEAT_URL` | *(unset)* | `http(s)` URL to ping after a verified restore. Treat it as a credential. `restore_check --scheduled`, which is what the timer runs, refuses to start without it |
| `BACKUP_SPOOL_DIR` | `/var/lib/nescio-backup` | must exist and be writable. Holds as many dumps as retention keeps, plus one being checked |
| `BACKUP_RETENTION_DAILY` | `14` | `1..90` |
| `BACKUP_RETENTION_WEEKLY` | `8` | `0..52` |

`VECTOR_WEIGHT` and `EMBEDDING_DIMENSION` from the API's configuration also apply to the restore
check, which runs the production search: `VECTOR_WEIGHT=0` fails it, and a non-default width has to
be set in the unit's environment as well.

### Setting it up on the VPS

The scripts run on the host, against the database through a loopback-only port, and need
`pg_dump`/`pg_restore` no older than the server. The app image is not involved. `compose.yaml`
publishes no database port, so add an override, and give `POSTGRES_PASSWORD` a real value in `.env`
first, because its default is only safe while the port is not published:

```yaml
# compose.override.yaml
services:
  db:
    ports:
      - "127.0.0.1:5432:5432"
```

```bash
sudo apt install postgresql-client-17 rclone
sudo useradd --system --home-dir /nonexistent --shell /usr/sbin/nologin nescio-backup

# The checkout the unit runs. --no-dev: tests are not needed on the host.
sudo git clone https://github.com/noctua84/nescio-memory /opt/nescio-memory
cd /opt/nescio-memory && sudo uv sync --locked --no-dev

# An S3-compatible store, and a crypt remote over it. `rclone config` asks for
# both; keep the crypt passwords somewhere other than this VPS. Without them the
# backups cannot be read by anyone, you included.
sudo install -d -m 0755 /etc/nescio
sudo rclone --config /etc/nescio/rclone.conf config
sudo chown nescio-backup: /etc/nescio/rclone.conf && sudo chmod 0400 /etc/nescio/rclone.conf

sudo install -m 0600 /dev/null /etc/nescio/backup.env
sudoedit /etc/nescio/backup.env
```

```bash
# /etc/nescio/backup.env
DATABASE_URL=postgresql://nescio:password@127.0.0.1:5432/nescio_memory
RESTORE_CHECK_DATABASE_URL=postgresql://nescio:password@127.0.0.1:5432/postgres
BACKUP_RCLONE_REMOTE=secret:nescio
BACKUP_HEARTBEAT_URL=https://hc-ping.com/your-check-uuid
RCLONE_CONFIG=/etc/nescio/rclone.conf
```

```bash
sudo cp deploy/systemd/nescio-backup.service deploy/systemd/nescio-backup.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nescio-backup.timer
sudo systemctl start nescio-backup.service   # one run now, to see it work
journalctl -u nescio-backup.service -o cat
```

A good run ends like this, with your own numbers:

```text
backup uploaded dump=nescio-20261104T182659Z.dump size_bytes=94297717 clients=3 notes=4000 passages=20000 pruned_spool=0 pruned_remote=0
restore measured dump=nescio-20261104T182659Z.dump size_bytes=94297717 clients=3 notes=4000 passages=20000 pg_restore_s=11.9 elapsed_s=15.1
restore verified heartbeat=sent
```

`TimeoutStartSec=6h` in the unit bounds a whole run. Raise it if the measured time gets close.

### Runbook

**Back up now, or check the newest backup by hand.** Both read `/etc/nescio/backup.env` through the
unit; to run them directly, export its contents first.

```bash
sudo systemctl start nescio-backup.service       # backup, restore check, heartbeat
uv run python -m scripts.backup                  # exit 0 uploaded, 1 failed, 2 invalid settings
uv run python -m scripts.restore_check           # exit 0 verified, 1 failed, 2 refused
```

**Recover.** This restores into a new database and leaves the current one alone until you switch.
The restoring role must be able to create the `vector` extension.

```bash
export RCLONE_CONFIG=/etc/nescio/rclone.conf PGPASSWORD=password
SERVER=postgresql://nescio@127.0.0.1:5432

# 1. Choose a backup; names sort by time. Listing through the crypt remote decrypts them.
rclone lsf secret:nescio --include '*.dump' | sort
BACKUP=nescio-20261104T182659Z

# 2. Fetch it and check it against its manifest. The two sums must be equal.
rclone copyto "secret:nescio/$BACKUP.dump" "./$BACKUP.dump"
rclone copyto "secret:nescio/$BACKUP.manifest.json" "./$BACKUP.manifest.json"
sha256sum "$BACKUP.dump"
grep sha256 "$BACKUP.manifest.json"

# 3. Restore. --exit-on-error, so a failure stops here instead of leaving half a database.
psql "$SERVER/postgres" -c 'CREATE DATABASE nescio_recovered TEMPLATE template0'
pg_restore --no-owner --no-privileges --exit-on-error \
  --dbname="$SERVER/nescio_recovered" "$BACKUP.dump"

# 4. Verify: these must match the manifest, and the index must say t.
psql "$SERVER/nescio_recovered" \
  -c 'SELECT version_num FROM alembic_version' \
  -c 'SELECT client_name, count(*) FROM notes GROUP BY 1 ORDER BY 1' \
  -c 'SELECT client_name, count(*) FROM passages GROUP BY 1 ORDER BY 1' \
  -c "SELECT indisvalid FROM pg_index WHERE indexrelid = 'passages_embedding_idx'::regclass"
grep -A 12 '"clients"' "$BACKUP.manifest.json"

# 5. A restore has no planner statistics yet, and search plans suffer until it does.
vacuumdb --analyze-only --dbname="$SERVER/nescio_recovered"
```

Then point `DATABASE_URL` at `nescio_recovered` and restart the API, or stop it and rename the
databases. `GET /ready` must report the database `ok`; if the checkout is newer than the backup, run
`alembic upgrade head` first. Delete the local dump afterwards: it is your notes, unencrypted.

**When the alert fires** (`journalctl -u nescio-backup.service -o cat`):

| The journal says | Meaning |
| --- | --- |
| nothing recent | the timer or the VPS is down: `systemctl list-timers nescio-backup.timer` |
| `pg_dump failed with exit code N` | repeat `pg_dump -Fc --file=/tmp/x.dump "$DATABASE_URL"` by hand to see why. A client older than the server is refused |
| `rclone … failed with exit code N` | repeat with `rclone lsf secret:nescio -v`. Any non-zero exit fails the run, the "some files not transferred" codes included |
| `pg_restore failed with exit code N` | repeat the restore step above to see why. The dump is suspect until it passes |
| `does not match its manifest's sha256`, `counts differ`, `alembic revision` | **the backup is not what it claims to be.** Take a new one now and do not rely on this one |
| `HNSW index … is missing` or `is not valid` | the restore did not reproduce the index. Check the restore server's `maintenance_work_mem` and disk |
| `top hit` | the embeddings or the index did not survive, **or**, once, HNSW's own limit: part of the graph can be unreachable (see [Current limitations](#current-limitations)), and the probe is one passage. Measured on the synthetic data under [Measured restore time](#measured-restore-time), the share of passages the index could not find from their own embedding was 13 of 20,000 (0.07%) at 20,000 passages and 677 of a 20,000 sample (3.4%) at 100,000, so this rises with the corpus. Run the check again, since each restore builds the index anew. If it fails twice, treat the backup as bad |
| `the heartbeat did not land` | the backup verified, but the ping failed three times. The run counts as failed, and you get the alert. Check the VPS's egress and the URL |

### Measured restore time

From real runs of the unit on 2026-11-04, under systemd, through a real crypt remote over an S3
protocol store. **This is seeded data on one developer machine, not an estimate for production.** The
vectors are synthetic and 1024-wide, and all the containers shared a 24-core, 31 GB Docker VM with
PostgreSQL's defaults (`maintenance_work_mem` 64 MB, `max_parallel_maintenance_workers` 2). Take your
own number from your own journal, where every run reports `pg_restore_s` and `elapsed_s`.

| Passages | Notes | Database | Dump | `pg_restore` | Restore check (fetch, restore, verify) | Whole unit |
| --- | --- | --- | --- | --- | --- | --- |
| 20,000 | 4,000 | 299 MB | 94 MB | 11.4 s | 13.4 s | 34 s |
| 100,000 | 20,000 | 1.5 GB | 471 MB | 117 s | 121 s | 204 s |

Five times the data took ten times as long to restore. The HNSW index is rebuilt during the restore and
dominates the time, and once the graph outgrows `maintenance_work_mem` the build slows sharply, so give
the restoring server more of it, and more `max_parallel_maintenance_workers`, before a large restore.
`pg_restore --jobs` helps across indexes rather than within one build. `embedding_degenerate` is a
generated column and is recomputed by the restore, not copied.

### Replacing the timer with a Kubernetes CronJob

Not built; this is what moves when the service leaves the VPS. A CronJob running the same two commands
in order replaces the unit, with `concurrencyPolicy: Forbid` for what `Type=oneshot` gives, and
`activeDeadlineSeconds` for `TimeoutStartSec`. It needs an image with `pg_dump`, `pg_restore` and
`rclone`, because the app image deliberately has neither, and an `emptyDir` sized for one dump as
the spool, since a pod's spool does not survive and the remote then holds the only copies. The
heartbeat and the scratch database's rules do not change.

## Development

```bash
uv sync --locked                      # install / refresh the environment
uv sync --locked --extra local-embeddings  # only if using the in-process backend
uv lock                               # after changing dependencies in pyproject.toml
uv run python export_openapi.py       # regenerate openapi.json + openapi.yaml
uv run uvicorn app.main:app --reload --port 8000  # dev server (from the repository root)
uv run pytest                         # integration suite (needs Docker)
uv run alembic upgrade head           # apply migrations
uv run python -m scripts.check_embedding_dimension  # does the model's width match EMBEDDING_DIMENSION?
```

Run the dev server **from the repository root**: `app` is imported from the working directory, and
`.env` is resolved relative to it as well. Interactive docs are then at
<http://localhost:8000/docs>. This path needs steps 1 to 3 of the [Quick start](#quick-start).

Five things are enforced by CI:

- **The test suite must pass.** `uv run pytest` runs the integration suite against a real
  PostgreSQL + pgvector container on every push and pull request. No count is quoted here because
  it goes stale; `uv run pytest --collect-only -q` reports it.
- **`uv.lock` must stay in sync with `pyproject.toml`.** If you add or change a dependency, run
  `uv lock` and commit both files.
- **The committed OpenAPI spec must match the code.** If you touch a route, a `Form(...)` field, or
  a Pydantic model, run `export_openapi.py` and commit the regenerated `openapi.json` and
  `openapi.yaml`.
- **The `local-embeddings` extra must still resolve.** A `--dry-run` sync verifies it against the
  lockfile on every run without downloading torch, so the opt-in backend cannot rot unnoticed.
- **The container image must build and boot.** [`docker.yml`](.github/workflows/docker.yml)
  builds the image on every pull request and starts `db`, `migrate` and `app` (without Ollama),
  asserting that `/ready` reports the database `ok`.

### Re-ingesting a corpus

`scripts/reingest.py` walks a directory tree and `POST`s each Markdown file to
`/api/v1/ingest`:

```bash
uv run python -m scripts.reingest --root ~/notes --api-key nm_your_key_here \
  --url http://localhost:8000 --scope my-corpus
```

Two files are **excluded by default**, no matter what: `**/MEMORY.md` and
`**/learning-log.md`. Both are generated artifacts — `MEMORY.md` is an index of other notes'
descriptions and `learning-log.md` is a large promotion ledger — and both out-rank the original
notes on exact-keyword FTS, which is a retrieval-quality problem, not a convenience. `--exclude`
is **additive** to these defaults, so adding an unrelated exclusion never silently re-admits
them; `--no-default-excludes` is the explicit, visible way to drop them. `--force` passes
`force: true` on every file, so an unchanged file is re-split and re-embedded rather than
skipped — useful after a model change the pipeline version does not see. `--dry-run` lists what
would be posted and makes no HTTP call. `--url` must point at an allowed host (`localhost`,
`127.0.0.1`, `::1` by default); extend the allowlist with the comma-separated
`NESCIO_ALLOWED_HOSTS` environment variable.

### Measuring retrieval quality

`scripts/eval_queries.py` runs a fixed list of queries against `/api/v1/search` and reports how
many land the expected note in the top results — the harness for measuring hybrid search quality.
Run it with `VECTOR_WEIGHT=0` on the server for the keyword-only baseline to compare against. It
exits `2` (JSON `"valid": false`) if any response is `degraded`, because a keyword-only answer is
not a hybrid result.

```bash
uv run python -m scripts.eval_queries --api-key nm_your_key_here \
  --queries tests/fixtures/golden_queries.yaml
```

Queries live in `tests/fixtures/golden_queries.yaml`: each entry has a `query` string and an
`expect_path_or_uri` (a single path/URI, or a list when more than one note could legitimately
answer it). The file ships with three example queries; populate it with your own 15–20 real
queries against your corpus. The gate is **15 of 20 queries hitting the expected note within the
top 5 results** — below `gate.minimum_queries` (15) the gate is void and the script exits `3`
rather than reporting a failure. Record the resulting ratio in the spec doc's baseline table.
`--url` must point at an allowed host (`localhost`, `127.0.0.1`, `::1` by default); extend the
allowlist with the comma-separated `NESCIO_ALLOWED_HOSTS` environment variable.

Releases are automated by [release-please](https://github.com/googleapis/release-please) from
Conventional Commits (`feat:`, `fix:`, `chore:`, …). Merging the release PR it maintains bumps the
version and publishes a GitHub Release — do not edit versions by hand.

## Current limitations

This is a PoC. Known rough edges, roughly in order of how much they matter:

- **Search recall is bounded, not guaranteed.** pgvector's HNSW index is approximate, and the
  vector arm filters by `client_name`, `scope` and `embedding_degenerate` after the index produces
  candidates. The search sets `hnsw.iterative_scan = strict_order` so the index keeps looking until
  it has the candidates it was asked for, which fixes the worst case (a client holding a few rows of
  a large table getting one result back). It remains approximate — HNSW's graph construction is
  randomized and part of the graph can be unreachable for any given query, so the vector arm can
  still come back short, and a client holding a very small share of a very large table can still
  receive fewer candidates than asked for rather than an error. `hnsw.max_scan_tuples` is a safety
  cap on work, not the limit on recall; raising it does not lengthen a short result.
- **Hybrid search has no relevance floor.** The vector arm always returns its nearest neighbours,
  so a search fills up to `top_k` whenever that many passages exist and trailing results can be weak
  matches. A result with `fts_rank: null` is a semantic-only match. `score` is comparable only
  within one response.
- **`degraded: true` responses are keyword-only.** When the query cannot be embedded because the
  embedding backend is briefly unavailable (a timeout, a connection failure, or HTTP 408, 429, 500,
  502, 503 or 504), search answers from the full-text arm alone instead of failing. A malformed
  backend response or a misconfigured backend is not transient and still returns an error: every
  other non-2xx status (a 404 for a model that is not pulled, a 501 for a non-embedding model, a
  redirect such as the 307 from a trailing-slash `OLLAMA_URL`) returns a `500` with no `Retry-After`. Treat `degraded: true` as a weaker result,
  not an error; `scripts/eval_queries.py` refuses to score one.
- **Langfuse keys in `.env` are ignored** when you run with `uv`. `@observe` relies on the SDK
  reading `LANGFUSE_*` from the process environment, and pydantic-settings does not export `.env`
  values into it. Export the keys in your shell or service manager, or tracing silently stays off.
  This does not apply to [Docker Compose](#5-run-with-docker-compose), which exports `.env` into the
  container's environment.
- **Ingestion retries transient embedding failures, within a fixed bound.** Passages are embedded
  through Ollama's `/api/embed` in groups of `EMBED_BATCH_SIZE`. A call that fails with a timeout,
  a connection error or HTTP 408/429/500/502/503/504 is retried with exponential backoff and full
  jitter (`EMBED_RETRY_BASE_DELAY_S`, doubling, capped at `EMBED_RETRY_MAX_DELAY_S`; a `Retry-After`
  from a 429 or 503 is honoured up to that cap), at most `EMBED_RETRY_ATTEMPTS` tries per group.
  A bad response or a misconfiguration is never retried. The retries are drawn from one pool of
  `EMBED_RETRY_BUDGET` per request, shared by all groups, so a large note cannot retry once per
  group. Ingest holds a row lock while it embeds, so the worst case is stated: the lock is held
  at most `EMBED_RETRY_BUDGET × (30 s + EMBED_RETRY_MAX_DELAY_S)` longer than a request that never
  retries (defaults: `2 × (30 + 4) = 68 s`), whatever the note's size. When the budget runs out
  the request fails with `503` as before and the client re-sends it. A search retries at most
  once and only inside `QUERY_EMBED_TIMEOUT_S`; if that is not enough it degrades to keyword-only
  as before.
- **There is no list or export route.** No enumeration, and no fetch-by-path (deleting by path
  exists, reading by path does not): a note is readable by id (`GET /api/v1/notes/{id}`) or as
  passages through a search that matched it. This is an
  architectural property and **not a security boundary**: there is no rate limiting, so bulk
  retrieval is bounded by query effort rather than refused. Recovering the corpus itself is a
  `pg_dump`/restore concern, not an API one: see [Backup and recovery](#backup-and-recovery). A note
  changes only by re-ingesting it whole.
- **Changing the FTS configuration or weights forces a full re-ingest**, because `tsv` is
  materialized at ingest time, not computed on the fly.
- **A note whose source file was deleted is not removed automatically.** A client removes it
  with `DELETE /api/v1/notes/{id}` or `DELETE /api/v1/notes?scope=…&path_or_uri=…`, or retires a
  whole scope with `DELETE /api/v1/scopes`; reconciling notes against a source tree is not
  something the service does.

Contributions addressing any of the above are welcome.

## License

MIT — see [LICENSE](LICENSE).
