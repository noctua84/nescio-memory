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
      "similarity": 0.8123,
      "context_ref": null
    }
  ],
  "contexts": []
}
```

| Field            | Type             | Default  | Description                            |
| ---------------- | ---------------- | -------- | -------------------------------------- |
| `query`          | string           | —        | text to search for (required); blank or whitespace-only after stripping is rejected with `422` -- surrounding whitespace is otherwise kept and forwarded unchanged |
| `top_k`          | int              | `5`      | maximum number of results; must be between 1 and 50, otherwise `422` |
| `repo_filter`    | string \| `null` | `null`   | restrict results to one `repo_name`    |
| `context`        | string           | `"none"` | `"none"`, `"neighbors"` or `"document"` — how much surrounding text to return |
| `context_chunks` | int              | `1`      | chunks either side of a hit, 1–5; only meaningful for `"neighbors"` |

Each result carries `content` (the matched chunk), `metadata` (`file_name`, `relative_path`,
`chunk_index`), `similarity` and `context_ref`.

#### Context expansion

A chunk is a positional 1000-character window, so a hit routinely begins and ends mid-thought.
`context` asks for what surrounded it: `"neighbors"` expands each hit by `context_chunks` chunks
either side, `"document"` expands to the whole note the hit came from. `"none"` is the default and
runs no extra query at all, so existing callers are unaffected.

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: nm_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{"query": "where do embeddings come from?", "top_k": 3, "context": "neighbors", "context_chunks": 2}'
```

```json
{
  "results": [
    {
      "content": "...the chunk that matched...",
      "metadata": { "relative_path": "docs/architecture.md", "chunk_index": 4 },
      "similarity": 0.8123,
      "context_ref": 0
    },
    {
      "content": "...another chunk, same note...",
      "metadata": { "relative_path": "docs/architecture.md", "chunk_index": 6 },
      "similarity": 0.7456,
      "context_ref": 0
    }
  ],
  "contexts": [
    {
      "repo_name": "nescio-memory",
      "file_path": "docs/architecture.md",
      "content": "...expanded text, overlap stripped at every join...",
      "chunk_index_from": 2,
      "chunk_index_to": 8,
      "covers": "neighbors",
      "exact": true,
      "note": null
    }
  ]
}
```

Expanded text lives in `contexts`, not inline on each result, and `context_ref` is the index of the
region containing that result. Regions are deduplicated and merged: two hits in the same note share
one region rather than repeating its text, and windows that overlap or touch become a single region
rather than two with a shared middle. `context_ref` is `null` when there is no region for a hit.

| Context field      | Description                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| `repo_name`        | repository the region's note belongs to                                  |
| `file_path`        | relative path of the region's note                                       |
| `content`          | the expanded text, with the chunks' shared overlap stripped at each join; `null` when the stored chunks admit no safe join |
| `chunk_index_from` | lowest chunk index included                                              |
| `chunk_index_to`   | highest chunk index included                                             |
| `covers`           | the `context` mode the region was requested under                        |
| `exact`            | whether the region provably reproduces that span of the original note    |
| `note`             | why it is not exactly what was asked for, when it is not                 |

Consecutive chunks share `chunk_overlap` characters, so a region is joined with that overlap
stripped rather than concatenated — and with the window each row records having been cut with at
ingest, not whatever the service is configured with now. Reassembling a note cut at 1000/200 under
a guessed `CHUNK_OVERLAP=0` returns text 23% longer than the original while passing every
consistency check, which is why the window is read from the rows.

**Context is only ever returned for a note a query matched.** There is no list route, no
fetch-by-path and no way to retrieve a note you could not find by asking for it; expansion is
strictly more of what `search` already returned. See
`docs/superpowers/specs/2026-10-03-search-context-expansion-design.md`.

**Expansion degrades rather than erroring.** It is additive, so it can only ever fail to add, and
no problem in it turns a successful search into a non-200:

| Situation                                                        | Result                                        |
| ---------------------------------------------------------------- | --------------------------------------------- |
| a chunk was dropped at ingest, leaving a hole in the index sequence | the region stops at the hole rather than splicing across it, and says so in `note` |
| a note's rows disagree about the window they were cut with       | no region for that note; its results get `context_ref: null` |
| more than `MAX_CONTEXT_DOCUMENTS` notes were matched            | only the best-ranked notes are expanded; the rest get `context_ref: null`, and the last included region's `note` says the limit was reached |
| a note's range is measured as too large for `MAX_CONTEXT_CHARS`  | that range is not fetched at all; the results referencing it get `context_ref: null`, and the last included region's `note` says the budget was reached before fetching — a note that also has another, smaller range admitted still gets context from that range |
| the response's character budget (`MAX_CONTEXT_CHARS`) is reached | lower-ranked regions are omitted; their results get `context_ref: null`, and the last included region's `note` says the budget was reached |
| anything else unexpected                                         | the search returns normally with `contexts: []` and every `context_ref: null` |

Regions are built in descending order of the best similarity among the results referencing them,
so the budget is spent on the context most likely to be wanted.

**The response never exceeds `MAX_CONTEXT_CHARS`; the database fetch behind it can, by design.**
Each stored chunk is a full `CHUNK_SIZE` window on disk, but the pre-flight (`admit_ranges` /
`_region_chars_lower_bound` in `app/core/context.py`) charges only about one step
(`CHUNK_SIZE - CHUNK_OVERLAP`) per chunk against the budget — a deliberately conservative LOWER
bound, so the check never withholds a region `build_regions` would in fact have admitted. The
fetch can therefore approach `MAX_CONTEXT_CHARS * CHUNK_SIZE / (CHUNK_SIZE - CHUNK_OVERLAP)`, plus
a couple of `CHUNK_SIZE`s per admitted range (the estimate's own margin and a tail-probe chunk
every range fetches just past its boundary and the builder never returns) and the tail-probe
rows themselves — about 1.25× the response budget at the defaults (1000/200), and as much as 10×
at 1000/900. The number of admitted ranges is bounded by how many hits were matched (`top_k`), not
by `MAX_CONTEXT_DOCUMENTS` — in `neighbors` mode one document can contribute several ranges.
Worked example: a 100-chunk note at the defaults charges the budget 78,400 characters but fetches
100,000. Size memory and statement timeouts for the fetch, not just the response.

**Both bounds apply before the rows are read.** `MAX_CONTEXT_CHARS` on its own is spent on text
that has already been transferred, so `context: "document"` over 50 matched notes would fetch all
of them whatever their size and then discard most of it. Expansion therefore first keeps the
best-ranked `MAX_CONTEXT_DOCUMENTS` notes, then asks the database for the size of what it is about
to read and admits ranges against the same character budget — deliberately underestimating, so a
range whose rows are intact is never withheld for having been under-estimated. A range whose rows
disagree about their window, have a hole in their index sequence, or are stored twice under the
same `chunk_index` can still lose context even though the real text would have fitted; the note on
the last region says so when it does. Whatever a bound held back is explained in the
`note` of the last region that did fit, which is the only place adjacent to the cut; if nothing
fits at all, `contexts` is empty and every `context_ref` is `null`.

### `/api/v2` — notes and passages

> **Status: early.** `/api/v2` is a second, parallel storage model (`notes` + `passages`)
> alongside the untouched `/api/v1` (`learnings`). There is no data migration between them —
> see [Current limitations](#current-limitations). Design rationale and the retrieval-quality
> baseline live in
> [`docs/superpowers/specs/2026-10-29-notes-passages-slice-1-design.md`](docs/superpowers/specs/2026-10-29-notes-passages-slice-1-design.md).

Every `/api/v2` route requires the same `X-API-Key` header as `/api/v1`, and is tenant-scoped the
same way.

#### `POST /api/v2/ingest`

Stores one note as JSON (not `Form(...)` — there is no 1 MiB per-field byte limit here, only
`MAX_CONTENT_CHARS`). The note is split into passages, embedded, and both are written in one
transaction.

```bash
curl -X POST http://localhost:8000/api/v2/ingest \
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

#### `GET /api/v2/notes/{id}`

Returns one note by id, scoped to the calling client — a note belonging to another client is
**indistinguishable** from an id that does not exist at all: both are the same `404`.

```bash
curl http://localhost:8000/api/v2/notes/42 -H "X-API-Key: nm_your_key_here"
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

#### `DELETE /api/v2/notes/{id}`

The erasure path: removes a note and, by cascade, every one of its passages. Returns `204` with
no body on success, and the same indistinguishable `404` as `GET` for a missing or foreign id.
This is the **only** way to remove a note — nothing reconciles notes against a source tree, so a
client that stops holding a file must call this itself.

```bash
curl -X DELETE http://localhost:8000/api/v2/notes/42 -H "X-API-Key: nm_your_key_here"
```

#### `POST /api/v2/search`

**FTS only in this release** — there is no vector search on this path yet, and a result's `rank`
is not comparable to `/api/v1`'s `similarity`. Results are **passages**, each carrying its note's
identity, description and provenance.

```bash
curl -X POST http://localhost:8000/api/v2/search \
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
      "rank": 0.642
    }
  ]
}
```

| Field         | Default | Description                                                           |
| ------------- | ------- | ----------------------------------------------------------------------- |
| `q`           | —       | also accepted as `query` (required); parsed with `websearch_to_tsquery` semantics — quoted phrases, `OR`, leading `-` to exclude a word |
| `top_k`       | `5`     | maximum number of results, 1–50, otherwise `422`                        |
| `repo_filter` | `null`  | restrict to notes with that `scope`                                     |

Results are ranked by `ts_rank_cd` over three FTS weights: `A` (note name + heading path), `B`
(passage content) and `C` (note description). `description` is deliberately weighted below
`content`: it is note-level and duplicated onto every passage of the note, so at the same weight
as `content` it would score every passage of a long note identically and could outrank a passage
that genuinely discusses the term — the "returns a page to read again" failure this search exists
to avoid. A query that reduces to an empty tsquery (a stop word alone, or only punctuation) is not
an error — it returns `{"results": []}` with a `200`.

## Configuration

All settings are read from `.env` via [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/);
see [`.env.example`](.env.example) for the annotated template. Unknown keys are ignored.

| Variable              | Default                                    | Description                                          |
| --------------------- | ------------------------------------------ | ---------------------------------------------------- |
| `DATABASE_URL`        | *required*                                 | PostgreSQL connection string; a plain `postgresql://` uses the bundled psycopg 3 driver |
| `DB_CONNECT_TIMEOUT_S` | `10`                                       | seconds per connection attempt (applies per resolved address); with `pool_pre_ping` a dead database adds this inside a request; must be `>= 2`, otherwise the app refuses to start |
| `STATEMENT_TIMEOUT_MS` | `5000`                                     | caps each statement, not the whole request; must be `1..60000`; a search with context runs up to three such statements (the vector search, the context measurement, the context fetch), so the worst-case wall clock is up to three times this value; **breaking in 0.5.0**: values above 60000 now refuse to start |
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
| `CHUNK_OVERLAP`       | `200`                                      | overlap between chunks — must satisfy `0 <= CHUNK_OVERLAP < CHUNK_SIZE`, otherwise the app refuses to start; the closer this is to `CHUNK_SIZE`, the larger the database FETCH behind a search response can run over `MAX_CONTEXT_CHARS` — see [Context expansion](#context-expansion) |
| `MAX_CONTEXT_CHARS`   | `100000`                                    | total characters of expanded context one search response may carry; a cap, not a target. The database FETCH behind a response can run well over this — up to roughly `MAX_CONTEXT_CHARS * CHUNK_SIZE / (CHUNK_SIZE - CHUNK_OVERLAP)` plus a couple of `CHUNK_SIZE`s per admitted range — see [Context expansion](#context-expansion). Must be `> 0`, otherwise the app refuses to start |
| `MAX_CONTEXT_DOCUMENTS` | `25`                                      | how many distinct notes one search response may expand, in rank order; does NOT bound the number of admitted ranges in `neighbors` mode, which is bounded by the number of matched hits instead — see [Context expansion](#context-expansion). Must be `> 0`, otherwise the app refuses to start |
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

`3d80eb22a17d` is the last revision with the 384-dim schema — `uv run alembic history` confirms
the chain is `3d80eb22a17d -> 57cfbdfa0d6a` (restores 1024) `-> 9439c6362bb3` (adds stored chunk
stats) `(head)`. Name the target revision explicitly, as above; do not use `alembic downgrade -1`
or any other relative offset from `head` — later migrations land between `3d80eb22a17d` and
`head`, so `-1` stops at the wrong place as soon as the chain grows. With the stock
`EMBEDDING_DIMENSION=1024` the `local` backend fails every embedding call, loudly, on the
dimension check — the mirror image of the defect in #36.

Pointing `LOCAL_EMBEDDING_MODEL` at a genuinely 1024-dimensional sentence-transformers model
would make one schema serve both, and is the obvious next step; it is not taken here because
nobody has run it. An unexecuted dimension claim in this file is what caused #36 in the first
place.

#### Changing the embedding width is blocked by the stored generated column

`learnings.embedding_degenerate` is a `STORED GENERATED` column computed as
`(embedding <=> embedding) = 'NaN'::float8` (added in `9439c6362bb3`). Postgres refuses
`ALTER COLUMN embedding TYPE vector(N)` while any generated column reads `embedding` — the
column's expression is bound to the old type. A width-change migration must, in order: drop
`embedding_degenerate`, run the `ALTER COLUMN ... TYPE vector(N)`, then re-add
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

### Re-ingesting a corpus

`scripts/reingest.py` walks a directory tree and `POST`s each Markdown file to
`/api/v2/ingest`:

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
would be posted and makes no HTTP call.

### Measuring retrieval quality

`scripts/eval_queries.py` runs a fixed list of queries against `/api/v2/search` and reports how
many land the expected note in the top results — the harness for the quality gate slice 2's
ranked-fusion work has to beat.

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
- **There is no enumeration route.** No list, no fetch-by-path, no export: content is reachable
  only through a query that semantically matched it, by design. Note that a hit *can* be expanded
  to its whole document (`context: "document"` returns the note byte-for-byte), so this is a
  guarantee about **discovery**, not about response size — you can read what you can find, and you
  cannot find what you have not matched. It is deliberately an architectural guarantee and **not a
  security boundary**: there is no completeness signal and no rate limiting, so bulk retrieval is
  bounded by query effort rather than refused. Recovering the corpus itself is a `pg_dump`/restore
  concern, not an API one. There is no `UPDATE` or `DELETE` route either — a document changes only
  by re-ingesting it whole.

- **`/api/v2/search` has no semantic/vector branch yet.** It is FTS only (`websearch_to_tsquery`
  + `ts_rank_cd`); vector search and rank fusion arrive in a later slice.
- **`/api/v1` and `/api/v2` are two separate, unconverted storage models.** `/api/v1` and the
  `learnings` table still exist and are unchanged, with no data migration to `/api/v2`: the
  existing `learnings` rows are overlapping 1000-character fragments with no frontmatter and
  cannot be converted into notes — moving to `/api/v2` means re-ingesting from source.
- **Changing the FTS configuration or weights forces a full re-ingest**, because `tsv` is
  materialized at ingest time, not computed on the fly.
- **A note whose source file was deleted is not removed automatically.**
  `DELETE /api/v2/notes/{id}` is the only erasure path; reconciling notes against a source tree
  is a follow-up, not something this slice does.

Contributions addressing any of the above are welcome.

## License

MIT — see [LICENSE](LICENSE).
