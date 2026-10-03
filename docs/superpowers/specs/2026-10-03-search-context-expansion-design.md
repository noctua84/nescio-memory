# Context expansion in `/api/v1/search`

**Date:** 2026-10-03
**Status:** Approved, implemented in this branch
**Supersedes:** an earlier egress design (a document list plus a read-back endpoint), rejected on
design grounds before merge — see "Why not an egress path" below.

## Context

`POST /api/v1/search` returns up to 50 chunks ranked by cosine similarity. A chunk is a
1000-character window of a source note, cut by a sliding window with 200 characters of overlap. The
window is positional, not semantic: it does not respect paragraph, sentence or section boundaries,
so a hit routinely begins and ends mid-thought. The caller gets the passage that matched and no way
to see what surrounded it.

That is a retrieval-quality problem, and the standard answer is to expand a hit — to its
neighbouring chunks, or to the whole note it came from. This adds that.

## Why not an egress path

The first attempt at this was a list endpoint plus a fetch-by-path read-back, justified as
recoverability: "data that cannot be read back out cannot be recovered." That was wrong twice over,
and it is recorded here so it is not re-proposed.

**Recoverability is an infrastructure concern, not an API concern.** The service runs on a VPS with
daily backups. `pg_dump` and a restore recover the corpus completely, faster, and without being
limited to what chunk reassembly can prove. And if the drive is gone, an HTTP read-back endpoint is
gone with it — it protects against nothing that backups do not already cover.

**A list endpoint changes what the service is.** `GET /documents` is an enumeration primitive:
list everything, fetch each one. That makes "sync the corpus locally and search it yourself" the
cheapest strategy available, at which point the embedding layer is optional and the service is a
document store with vectorization bolted on. It is a vectorized knowledge store. Semantic search is
the interface, not one of several.

So the rule this design holds to: **content is reachable only through a query that semantically
matched it.** There is no enumeration, no fetch-by-path, and no way to retrieve a note you could not
find by asking for it. Context expansion is strictly more of what search already returns.

### What that rule does and does not guarantee

It is an architectural guarantee, not a security boundary, and it is worth being exact about the
difference so the rule is not trusted for more than it does.

It does guarantee there is no *supported* way to bulk-read: no listing, no fetch-by-path, no cursor
over the corpus, and no way to discover a note's existence except by matching it. An agent cannot
learn what it has not seen, cannot ask for a note it cannot describe, and gets no completeness
signal — so "mirror the corpus locally and search it myself" is not a strategy the API offers.

It does not make exhaustive retrieval impossible. A caller issuing many broad queries with
`context: "document"` will accumulate notes, bounded per request by `max_context_chars`. That is
accepted, and it is not a leak: an API key identifies a tenant, every query is scoped to that
tenant's own rows, and a tenant reading its own notes is authorised to do so. The point of the rule
is that doing it means working against the grain of the API rather than along a path provided for
it — which is what keeps the service a vectorized knowledge store rather than a document store with
vectorization attached.

Rate limiting, quotas and per-key budgets are the tools for the volume question, and none of them is
in this change.

## The shape

`SearchRequest` gains two fields:

| Field | Default | Meaning |
|---|---|---|
| `context` | `"none"` | `"none"`, `"neighbors"`, or `"document"` |
| `context_chunks` | `1` | how many chunks either side, for `"neighbors"`; 1–5 |

`"none"` is the default, so every existing caller is unaffected.

`SearchResponse` gains a `contexts` sidecar, and each `SearchResult` a `context_ref` index into it:

```json
{
  "results": [
    { "content": "...the chunk that matched...", "metadata": {...}, "similarity": 0.81, "context_ref": 0 },
    { "content": "...another chunk, same note...", "metadata": {...}, "similarity": 0.74, "context_ref": 0 }
  ],
  "contexts": [
    {
      "repo_name": "nescio-ai",
      "file_path": "docs/retrieval.md",
      "content": "...expanded text, overlap stripped...",
      "chunk_index_from": 2,
      "chunk_index_to": 6,
      "covers": "document",
      "exact": true,
      "note": null
    }
  ]
}
```

### D1. A sidecar, not a field on each result

Two hits in the same note under `context: "document"` would otherwise carry the same text twice.
With `top_k` up to 50 that is a response several times larger than it needs to be, and it wastes the
thing context expansion exists to serve — the model's context window — by putting the same passage
in the prompt repeatedly.

So expanded text is deduplicated. More than that, **overlapping regions are merged**: under
`"neighbors"`, two hits three chunks apart in the same note produce windows that overlap, and those
become one region rather than two with a shared middle. `context_ref` is how a result points at the
region containing it, and several results can point at the same one. Under `"none"`, `contexts` is
empty and every `context_ref` is `null`.

### D2. Expansion never fails the search

A search must not return an error because one hit's note has a problem. Every failure degrades to
less context, never to a non-200:

- a chunk dropped at ingest leaves a hole in the index sequence, so the region is **truncated at the
  hole** rather than spliced across it — splicing would join two passages that were never adjacent
- a note whose rows disagree about the window they were cut with yields no region for that note
- anything else unexpected yields no region for that note

In each case the affected region carries `exact: false` and a `note` saying what happened, or is
absent and the result's `context_ref` is `null`. The matched chunk itself is always returned
regardless — expansion is additive and can only ever fail to add.

This is a deliberate departure from the rejected egress design, which refused with `409` rather than
return an imperfect reconstruction. That was right for an archival read ("is this byte-identical to
what was ingested?") and is wrong here. The question a retrieval caller is asking is "what surrounds
this passage?", and slightly-short context is a useful answer where an error is not. `exact` is
still reported, so a caller that cares can tell.

### D3. Overlap is stripped, and the window comes from the rows

Consecutive chunks share `chunk_overlap` characters. Concatenating them duplicates that overlap at
every join, which is both wrong and actively harmful in a prompt. `app/core/reassembly.py` strips
it, and its invariants are what make the result trustworthy.

The window used is the one recorded in each row's `metadata` at ingest (`chunk_size`,
`chunk_overlap`), not whatever the service is configured with now. This matters: reassembling with a
guessed window silently duplicates the overlap at every join. Measured during the rejected design's
review, a note ingested at 1000/200 and reassembled at `CHUNK_OVERLAP=0` came back 23% longer than
the original and *passed* every consistency check, because the check had nothing to compare. Rows
ingested before the window was recorded fall back to current configuration and cannot claim
`exact: true` where that fallback is unverifiable.

### D4. One query, not one per hit

The chunks needed for every region are fetched in a single query — the union of the required
`(repo_name, file_path)` sets, filtered by chunk index range — and the regions are assembled in
memory. A per-hit query would mean up to 50 round trips on one search.

### D5. A character budget, filled in rank order

`context: "document"` with 50 hits in 50 distinct notes could in principle return 50 ×
`MAX_CONTENT_CHARS`. Regions are built in descending order of the best similarity among the results
referencing them, until `settings.max_context_chars` is reached. Results whose region did not fit
get `context_ref: null`; the region that was cut carries a `note` saying so if it was partially
included. The budget defaults to 100,000 characters — roughly 25 notes at this corpus's average
size — and is a cap, not a target.

### D6. Tenant scoping

The context query is a **new** query and filters on `client_name` independently, through
`LearningRepository`, exactly as `search` does. It does not treat the `(repo_name, file_path)` of a
hit as sufficient authorisation to read those rows, even though the hit itself came from a
client-scoped search: a filter that is correct only because an earlier filter was correct is one
refactor away from being wrong.

Covered by `tests/test_isolation.py` in its existing strongest form — two requests differing only in
the key presented — plus the case specific to this feature: two tenants holding the same
`repo_name` and `file_path`, each expanding a hit and seeing only their own note.

## Kept from the rejected design

- `app/core/reassembly.py` and its invariants — now retrieval infrastructure rather than export
  machinery.
- The `chunk_size`/`chunk_overlap` provenance recorded in each row's `metadata` at ingest, which is
  what makes overlap stripping sound. JSONB, so still no migration.
- The int8 cast with a bounded digit guard on `chunk_index`, so a malformed value degrades instead
  of erroring.
- `chunk_text` no longer discarding an explicit `overlap=0` through a falsy `or`.

## Discarded

`GET /api/v1/documents`, `GET /api/v1/documents/content`, `app/schemas/documents.py`,
`LearningRepository.list_documents` / `count_documents`, and their tests. Along with them the
pagination envelope, `path_prefix`, the date filters, and the archival `exact` / `allow_partial` /
`window_source` response contract.

## Out of scope

- **Semantic chunking.** Expanding a positional chunk treats the symptom; cutting on paragraph or
  heading boundaries treats the cause. That changes ingest, invalidates every stored embedding, and
  is a much larger change.
- **Re-ranking the expanded region.** Context is returned as stored, not re-scored.
- **Any read path that does not start with a query.** By construction, per the rule above.
