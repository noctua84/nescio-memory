# Hybrid passage search, then removing `/api/v1` and `learnings` — slices 2 and 3

**Date:** 2026-10-31
**Status:** Approved, implemented on this branch (PR #82)
**Supersedes:** the "out of scope" list of `2026-10-29-notes-passages-slice-1-design.md`.
Slices 2 (vector search and rank fusion) and 3 (removing `/api/v1` and `learnings`) were
delivered together, in this order, on one branch.

## Context

Slice 1 added `notes` + `passages` beside the untouched `learnings` table and served them
under `/api/v2` with full-text search only. Two things were left open on purpose: a vector
branch on the passages path, and the removal of the old `learnings` model.

The old `/api/v1` learnings API was never deployed or used, so it carries no compatibility
obligation. That removed the usual reason to keep it alive, and left only one ordering
constraint: semantic search must exist on the passages path *before* the old path is deleted,
so the service is never left without it.

Full background: `.sisyphus/discovery/hybrid-search-v1-removal.md` (the discovery brief) and
`.sisyphus/plans/hybrid-search-v1-removal.md` (the plan, decisions P1–P9). The decisions below
restate what those two documents fixed; nothing here is new.

### What the service looks like after this change

- `POST /api/v1/ingest`, `POST /api/v1/search`, `GET /api/v1/notes/{note_id}` and
  `DELETE /api/v1/notes/{note_id}` serve the passages model. The operation ids lost their
  `V2` suffix: `ingestNote`, `getNote`, `deleteNote`, `searchPassages`.
- Search is one hybrid statement in `app/repositories/passage_search.py`, with the pgvector
  and timeout guards in `app/repositories/search_guards.py`: a full-text arm and a vector
  arm, each filtered by client and scope **before** its own `LIMIT`, fused with reciprocal
  rank fusion (RRF).
- Ingest embeds passages in batches through Ollama `/api/embed`.
- `learnings`, its API, its repository and the context-expansion code are gone.

## Design decisions

The D-numbers are local to this spec. "Brief" means
the discovery brief, "P*n*" a planning decision in the plan.

### D1. Vector search and fusion land before v1 is deleted

Brief, "Decisions already taken". Deleting `/api/v1` first would leave a window with no
semantic search. **Rejected:** removing v1 first and adding the vector branch in a later PR.

### D2. v1 is deleted outright, and `learnings` is dropped by a new migration

Brief, "Decisions already taken". v1 was never deployed or used, so there is no deprecation
window. The table is dropped by migration `3d67961c1596` (commit `4b05ec9`), not by editing
history. Its downgrade recreates an empty `learnings` table in its final shape, including both
stored generated columns and their indexes, so the revision chain stays reversible. Because
`config.py` ignores unknown environment variables (`extra="ignore"`), the removed settings
(chunk size, overlap, `MAX_CONTEXT_*`) are harmless if an operator leaves them set.

### D3. The passages API moves from `/api/v2` to `/api/v1`

Brief, "Decisions already taken". With v1 gone, the `v2` label named a version that no longer
had a predecessor. The prefix, operation ids, tags, schema module names and test names all
dropped the suffix. The rename is behaviour-critical in two places beyond the routers: the
body-size limit (`app/core/body_limit.py`, which matches on the prefix) and the scripts that
hard-code paths (`scripts/reingest.py`, `scripts/eval_queries.py`). Order mattered: v1 was
deleted **before** the rename, because both routers would otherwise have claimed the same
paths. The rename is its own commit.

### D4. This is a breaking release: 0.6.0

Removing endpoints and a table, and moving the prefix, is breaking. The deletion commit
(`4b05ec9`) and the prefix commit (`b7b52fb`) are both `feat!`, so release-please cuts 0.6.0
(`bump-minor-pre-major`). Plan P8: both land in the changelog; the deletion footer names the
removed learnings endpoints and table. CHANGELOG itself is release-please's to edit.

### D5. One SQL statement, fused with RRF, with a rank-based score

Brief, "Assessment" and "Analysis". Two arms each rank their own candidates with
`row_number()`, are combined with `UNION ALL` + `GROUP BY` on the passage id, and scored as
`sum(weight / (RRF_K + rank))`. Each result carries `score`, `fts_rank` and `vector_rank`
(rank within its arm, or `null` when the arm did not return it). Ordering is `score DESC`,
then passage id, so ties are deterministic.

- **Vector arm.** The inner query orders by cosine distance *only* and applies `LIMIT`; an id
  tiebreak inside it would defeat the HNSW ordering. The id tiebreak lives in the outer
  `row_number()`.
- **Full-text arm.** The slice-1 predicates (`websearch_to_tsquery('english')`, `ts_rank_cd`
  with A/B/C weights), ranked by `rank DESC, id`.
- **Candidates.** Each arm returns `max(SEARCH_CANDIDATES, top_k)` rows.
- **Metadata.** The note join happens once, after fusion, with the tenant predicate on both
  tables (slice 1's tenant-filter-at-top-level rule is unchanged).
- **Empty tsquery.** A query made only of stop-words yields an empty tsquery that matches
  nothing. It needs no special case: the vector arm still answers. A test pins this.
- **Query vector.** Passed as a bound parameter.

**Rejected: score normalisation.** `ts_rank_cd` is unbounded and depends on the query;
cosine similarity is bounded. Normalising one against the other is unstable. RRF uses only
ranks, which are comparable across arms by construction (brief, "External research" and
"Assessment").

**Rejected: a reranker.** At this corpus size its cost exceeds its benefit (brief,
"Assessment").

**Rejected: porting context expansion** (`context.py`, `reassembly.py`). Passages are already
bounded by headings, and `GET /api/v1/notes/{note_id}` returns the full body. Expansion
existed to repair chunking that no longer exists (brief, "Assessment").

### D6. Fusion knobs are settings, not API parameters

Brief, "Open questions" (orchestrator default) and plan P5. Each is read at call time:

| Setting | Default | Valid range |
|---|---|---|
| `SEARCH_CANDIDATES` | 50 | 1..1000 |
| `RRF_K` | 60 | >= 1 |
| `FTS_WEIGHT` | 1.0 | finite, >= 0 |
| `VECTOR_WEIGHT` | 1.0 | finite, >= 0 |
| `QUERY_INSTRUCTION` | `Given a web search query, retrieve relevant passages that answer the query` | `""` disables the prefix |
| `QUERY_EMBED_TIMEOUT_S` | 5.0 | 0 < x <= 30 |
| `EMBED_BATCH_SIZE` | 16 | 1..256 |

Exposing the knobs on the request would turn tuning into part of the public contract, and no
caller has a reason to set them per request. A validator rejects `FTS_WEIGHT` and
`VECTOR_WEIGHT` both being 0 (plan P3).

### D7. A weight of 0 drops its arm; `VECTOR_WEIGHT=0` means no query embedding at all

Plan P3 and brief, "Decisions after ANALYZE". `VECTOR_WEIGHT=0` removes the vector arm and the
route does not call the embedder, so the FTS-only baseline run does not depend on Ollama and
`degraded` is `false`. `FTS_WEIGHT=0` drops the full-text arm. This is what makes the baseline
measurable on the same build as the hybrid run (see the measurement protocol).

### D8. HNSW is kept honest: filters before `LIMIT`, iterative scan required

Brief, "Constraints and hazards" and "Analysis". Client and scope filters sit **inside** each
arm, before its `LIMIT`, because a post-filter on HNSW candidates starves the result set
(#12, #15, #74). The default `ef_search` of 40 is below the candidate count of 50, so
`hnsw.iterative_scan = strict_order` is required for correctness, not an optimisation. One
helper in `app/repositories/search_guards.py` sets `statement_timeout`, `hnsw.iterative_scan`
and `hnsw.max_scan_tuples` with `set_config(..., true)` and reads **all** of them back, raising
if any did not take effect, as v1 did. `ef_search` stays at the default (measured on v1:
iterative scan at 40 recovered the full `top_k` on 20 of 20 queries at half the buffers of
`ef_search = 200`). The pgvector >= 0.8.0 startup check already enforced this requirement.

### D9. `scope` is copied onto passages; the FK is 3-column

Brief, "Decisions after ANALYZE"; plan P1. Scope lives on `notes`, so filtering it in the
vector arm would be a post-join filter on HNSW candidates, the starvation class of D8. Migration
`17be352b1a4c` therefore adds `passages.scope`, backfilled from the owning note, the same
pattern `client_name` already follows.

The migration replaces the slice-1 foreign key. The shape is now:

```
fk_passages_note (note_id, client_name, scope)
    -> notes (id, client_name, scope)  ON DELETE CASCADE
```

backed by the new `uq_notes_id_client_scope` on `notes (id, client_name, scope)`. The slice-1
`uq_notes_id_client` is dropped, because nothing else references it. There is no `ON UPDATE`
action, so an attempt to change `notes.scope` (an identity column the upsert never writes)
fails loudly instead of leaving passages out of sync. The slice-1 spec described this key as
two columns `(note_id, client_name)`; that was correct for slice 1 and was superseded by this
migration. The downgrade restores the 2-column form.

### D10. Degenerate embeddings are excluded by a stored generated column

Brief, "Constraints and hazards"; "Analysis". `passages` had no equivalent of the
`embedding_degenerate` guard that `learnings` got in `9439c6362bb3` (#42, #53). Without it, a
zero-norm stored vector makes `<=>` return NaN, which would score as a perfect match in the
vector arm. The same migration (`17be352b1a4c`) adds `passages.embedding_degenerate` as a
`STORED` generated column, computed by Postgres from the stored vector (float4 storage can turn
a vector that Python accepted into a zero-norm one). The vector arm filters
`embedding_degenerate IS false` and rejects `NaN` distances. Adding a stored generated column
rewrites the table under an `ACCESS EXCLUSIVE` lock and rebuilds its indexes; no re-ingest is
needed. **Rejected:** a check in application code, which cannot see what float4 storage does to
the value.

### D11. Query-time embedding failures degrade; misconfiguration does not

See "The degraded-mode contract" below. Brief, "Decisions already taken" and "Decisions after
ANALYZE"; plan P2, P3 and P9.

### D12. Ingest embeds in batches through Ollama `/api/embed`

Brief, "Decisions already taken". Ingest holds the note's `FOR UPDATE` lock across the
embedding calls (slice-1 spec, transaction shape); batching shortens that hold.

- `OLLAMA_URL` keeps its name; the default is now `.../api/embed`. A startup validator in
  `app/config.py` fails fast on the legacy `/api/embeddings` URL and names the replacement. It
  does not rewrite the URL silently.
- `get_embeddings(texts)` sub-batches by `EMBED_BATCH_SIZE`, checks the returned count,
  validates every vector, and sends `"truncate": true` explicitly (plan P6). With `false`, an
  over-long passage would fail as an HTTP 400, and that 400 would otherwise be misread as a
  transient failure.
- Transport is layered (plan P4): `_fetch_batch` is the single raw transport for Ollama and the
  local backend; `get_embedding(t)` is `get_embeddings([t])[0]`; `get_query_embedding(q)` adds
  the instruction prefix and uses the short timeout.
- Ingest failures are unchanged: an embedding failure on ingest is still a 503.

### D13. The query instruction is on by default; passages are never prefixed

Brief, "Decisions after ANALYZE"; plan P5. Qwen3-Embedding is instruction-tuned on the query
side only. Queries are embedded as `Instruct: {instruction}\nQuery:{query}`; passages are
embedded unprefixed. Setting `QUERY_INSTRUCTION=""` disables the prefix, which a model without
an instruction format needs.

### D14. New test files carry no `v2` in their names

Plan P7, so the rename wave did not have to touch them.

## The degraded-mode contract

Search never returns 503 for a query-embedding outage.

- The response always carries a top-level boolean `degraded`. There is no reason field; the
  exception class name is logged, never the query (plan P2).
- The query is embedded **before** the first database statement, with a short timeout
  (`QUERY_EMBED_TIMEOUT_S`, default 5 s, separate from ingest's 30 s), so a dead backend does
  not hold a transaction open and an interactive search does not wait out the ingest timeout.
- On a **transient** failure the search runs the full-text arm alone and returns
  `degraded: true`. The discrimination is by exact class: the route catches
  `EmbeddingBackendError` and re-raises unless `type(exc) is EmbeddingBackendError`. The
  subclasses keep their existing mapping (plan P2): `EmbeddingBackendBadResponse`, which
  includes zero-norm, NaN and overflowing query vectors, is a 503; `EmbeddingBackendMisconfigured`
  is a 500.
- In degraded mode the full-text arm runs with effective weight `FTS_WEIGHT or 1.0`, because
  with weight 0 every score would be 0 (plan P3).
- With `VECTOR_WEIGHT=0` the embedder is not called at all; that is not degradation and
  `degraded` stays `false`.
- Ingest does **not** degrade: it still returns 503 with `Retry-After`.

Precedent: other hybrid-search systems degrade to keyword-only with a signal rather than fail
(brief, "External research").

### HTTP-status classification (P9)

A degradation must mean "retrying later may help". Without a rule for this, P2's exact-class
degrade would serve full-text results indefinitely while reporting `degraded: true`, as if the
condition were transient. So `_fetch_batch` classifies:

| Failure | Class | Effect |
|---|---|---|
| 4xx other than 408/429 (for example 400, 404) | `EmbeddingBackendMisconfigured` | 500, no `Retry-After`, logged at `error`; search fails |
| 408, 429, 5xx | `EmbeddingBackendError` | 503 + `Retry-After`; search degrades |
| Connection error, timeout | `EmbeddingBackendError` | 503 + `Retry-After`; search degrades |
| Local backend: model fails to load (bad `LOCAL_EMBEDDING_MODEL`) | `EmbeddingBackendMisconfigured` | search fails |
| Local backend: `encode()` runtime failure | `EmbeddingBackendError` | search degrades |

Why `Misconfigured` and not `BadResponse`: `BadResponse` maps to 503 with `Retry-After`, which
promises a retry that cannot help. A 404 means a model was never pulled or the Ollama build has
no `/api/embed`; a 400 means a non-embedding model. Each needs a deployment change, which is
exactly what `Misconfigured` means.

## Result semantics

- `score` is the RRF score. It is comparable only within one response, never across queries.
  This replaces slice 1's `rank` (`ts_rank_cd`), which was unbounded and query-dependent.
- `fts_rank: null` means the passage matched **semantically only**: it was in the vector arm's
  candidates and not in the full-text arm's. `vector_rank: null` is the mirror case.
- Hybrid search **always fills up to `top_k`**. Whichever arm has candidates contributes them,
  so a query with weak matches still returns `top_k` results.
- There is **no relevance floor**. A similarity threshold on the vector arm is out of scope.
  Clients judge relevance from `fts_rank` and `score` (a passage with `fts_rank: null` and a
  low `score` is a nearest-neighbour fill, not a confident hit).

## Slice boundary

This change ships: the `17be352b1a4c` migration (scope, degenerate flag, 3-column FK);
batched `/api/embed` ingest; hybrid RRF search with full-text-only degradation; removal of the
v1 learnings API and the `learnings` table (`3d67961c1596`); and the `/api/v1` prefix for the
passages API. Retrieval quality was measured with `scripts/eval_queries.py`, which exits
non-zero on any degraded response so a degraded run cannot be recorded as a result.

Explicitly **out of scope:** similarity thresholds on the vector arm; rerankers and score
normalisation; porting `context.py` / `reassembly.py`; API parameters for the fusion knobs;
`ef_search` tuning; deployment manifests; CHANGELOG edits.

## Measurement protocol

The decision to keep a vector branch rested on a measurement, as slice 1 required. The corpus
and the queries are private and are never committed: the real queries live in a local
uncommitted file passed to `scripts/eval_queries.py` with `--queries`, and only counts are
recorded in the repository.

1. **Two runs on one build.** Load the corpus once. Run the golden queries with
   `VECTOR_WEIGHT=0` first (FTS-only, no embedder involved), then with the defaults (hybrid).
   One build, one corpus, so the only difference between the runs is the vector arm.
2. **Counts only.** Record hits@5 out of the query total, the gate result and the number of
   degraded responses. No query text, no paths, no note names.
3. **Off-topic spot-check.** Against the hybrid server, send 2-3 queries about subjects the
   corpus does not contain. Confirm each returns up to `top_k` rows (there is no relevance
   floor) and that those rows mostly have `fts_rank: null`. Record only "observed as
   documented: yes/no", never the queries or the results.
4. **Optional third run.** Only if hybrid scores below FTS-only, run again with
   `QUERY_INSTRUCTION=""`, to test whether the instruction prefix is the cause.
5. **Decision point.** If hybrid scores **below** FTS-only, stop before merge. The options
   then are to ship with a `VECTOR_WEIGHT` default of 0, to tune `RRF_K`,
   `SEARCH_CANDIDATES` or the weights, or to revisit the query instruction. Record the
   decision here.

Any degraded response invalidates a run (the harness exits 2), because a "hybrid" number that
silently contains full-text-only responses would measure the wrong thing.

## Measured result

Both runs used the same build; there were 0 degraded responses in either.

| mode | corpus | notes | passages | queries | hits@5 | gate (15/20) |
|---|---|---|---|---|---|---|
| FTS-only (`VECTOR_WEIGHT=0`) | private corpus | 555 | 1631 | 20 | 7/20 | fail |
| hybrid (defaults) | private corpus | 555 | 1631 | 20 | 18/20 | pass |

**Verdict:** hybrid beats the FTS-only baseline by 11 queries and clears the gate, so hybrid
ships with the default weights (`FTS_WEIGHT=1.0`, `VECTOR_WEIGHT=1.0`, `RRF_K=60`,
`SEARCH_CANDIDATES=50`). The optional third run (`QUERY_INSTRUCTION=""`) was not needed, and
the decision point did not trigger. Off-topic spot-check: observed as documented (each
query returned `top_k` rows, all with `fts_rank: null`). The same table is recorded in the slice-1 spec, whose
baseline section this result completes.
