# `notes`/`passages` storage model — slice 1

**Date:** 2026-10-29
**Status:** Approved, implementing on this branch
**Supersedes:** nothing. `/api/v1`, `learnings`, `app/core/context.py`, `app/core/reassembly.py`
and `app/repositories/learning.py` are untouched by this slice.

## Context

`nescio-memory` started as a store for *learnings*: chunk a note, embed each chunk, keep
only the chunks. It was then pushed toward *document retrieval* (context expansion,
2026-10-03) without ever gaining a document entity. Data about each source document — its
name, description, type, provenance — ended up copied into `learnings.meta` JSONB on every
chunk, and an audit-driven hardening pass added defences around that JSONB rather than
around a typed model. The result: the ingest route is roughly 40 lines of logic, but the
machinery around it (`core/context.py`, `core/reassembly.py`, `repositories/learning.py`)
is close to 1,900 lines, most of which only makes sense read alongside the issues and audit
reports that produced it.

This service needs to hold both atomic learnings (small, mostly whole) and larger sources
(ADRs, design docs, external material) that benefit from being split, and to return the
specific learning or passage that matched a query — not a page of a document the caller has
to read again. Slice 1 adds a second storage model, `notes` + `passages`, alongside the
untouched `learnings` table, to make "document" a first-class, typed entity rather than
JSONB bolted onto a chunk.

Full background: `.sisyphus/discovery/notes-passages-model.md` (the discovery brief and the
architecture review outcome). The decisions below restate what that brief and the plan's
Context section (`.sisyphus/plans/notes-passages-slice-1.md`) already fixed; nothing here is
new.

### The schema

`notes` is one row per ingested unit (a memory note, an ADR, a design doc). Identity is
`UNIQUE(client_name, scope, path_or_uri)`. Typed columns carry what used to live in
per-chunk JSONB: `name`, `description`, `type`, `provenance_kind`, `provenance_date`,
`status`, `content_hash`, `pipeline_version`, the full `body`. `meta` JSONB remains, but only
for keys nothing queries.

`passages` is one row per structural split of a note's body: `note_id`, `client_name`
(copied, not just reachable via join), `ordinal`, `heading_path`, `content`, a weighted
`tsv`, and an embedding. `UNIQUE(note_id, ordinal)`.

The tenant filter in every query in this service is a single-table predicate on the table
being scanned — never a filter applied only after a join. That rule predates this slice
(`app/repositories/learning.py`) and is why `passages.client_name` is a copy of
`notes.client_name` rather than something reached through `note_id`: a composite foreign key
`(note_id, client_name) → notes(id, client_name)` (requiring `UNIQUE(id, client_name)` on
`notes`) lets `NoteRepository` put `client_name` directly in `passages`' own `WHERE` clause.
A post-join tenant filter would worsen the HNSW post-filtering issues already open against
this service (#12, #74); keeping the predicate single-table keeps that path clean for slice
2's vector branch.

## Design decisions

### D1. `notes.id` is `Integer` autoincrement, not a UUID

Matches `learnings.id` and `api_keys.id`. **Rejected:** a UUID primary key, which every
other table in this service avoids and which would make `notes` the odd one out for no
benefit — nothing in this slice or the next needs a globally-unique, pre-assignable key.

### D2. `passages` carries no timestamps

Passages are replaced wholesale on every content change; the note's `updated_at` is the
fact that matters, and a passage's own `created_at`/`updated_at` would just duplicate it on
every row. **Rejected:** per-passage timestamps mirroring `learnings`, which would be dead
weight — a passage is never updated in place, only deleted and reinserted with its note.

### D3. The skip decision is a `SELECT … FOR UPDATE` then an `INSERT … ON CONFLICT … RETURNING`, not `xmax = 0`

`NoteRepository.lock_note` + `NoteRepository.upsert_note` are two statements on purpose. A
single `INSERT … ON CONFLICT DO UPDATE … RETURNING` cannot tell the caller whether the row
pre-existed or what its *old* `content_hash` was: `excluded` is not visible in `RETURNING`,
and the row's `content_hash` there is already the post-update value. **Rejected:** the
`xmax <> 0` idiom, which is obscure and also non-zero for a row merely locked by another
transaction, not just one that already existed — it cannot distinguish "pre-existing" from
"concurrently contended." With the two-statement form, the only race (a concurrent inserter
landing between the `SELECT` and the `INSERT`) makes `lock_note` return nothing, the caller
treat the note as new, and the passages get rewritten redundantly — never wrongly. The row
lock from `lock_note`'s `FOR UPDATE` is what serialises concurrent ingests of the same note,
and it is held across the whole ingest, including the embedding calls (see D-transaction
below), not just across the two SQL statements.

### D4. The A- and C-weight FTS text is composed in Python, not concatenated in SQL

`app/core/pipeline.py:fts_lead_a`/`fts_lead_c` build the A and C strings before they reach
`NoteRepository.replace_passages`. **Rejected:** composing them in SQL with
`name || ' ' || heading_path`, because in Postgres `'x' || NULL` is `NULL` — any note missing
a `name` or `heading_path` (27 of the corpus's 574 notes have no frontmatter at all) would
silently blank that entire weight section rather than degrading to the part that is present.

### D5. `GET /api/v2/notes/{id}` returns `body` + `passage_count`, not passage texts

`body` already contains every passage's characters; shipping both the body and the passage
list would be redundant. The count must still be computed under the tenant filter (not
assumed from the request), so a cross-tenant count never leaks another tenant's passage
total. **Rejected:** returning the passage list alongside `body`, which duplicates data in
the response for no caller benefit in this slice — there is no v2 endpoint yet that needs a
note's passages individually without also having the body.

### D6. Embedding input is `name + heading_path + content`, not `name + description + body`

`app/core/pipeline.py:embedding_input` joins the non-empty parts with `\n`; `description` is
deliberately excluded from what gets embedded. **Rejected:** including `description`, which
is where D9 below lives instead — in the FTS document at weight C, not in the vector. A
note's description is a human summary, not necessarily phrased the way the passage itself
discusses the term, and folding it into every passage's embedding would pull every passage
of a note toward the same vector regardless of what that passage actually says.

### D7. `status` and `meta` are request-body inputs, not derived from frontmatter

ai-os is the authority on what a note's `status` is; the service stores what it is told,
rather than trying to infer status from the document body. **Rejected:** parsing `status`
out of frontmatter or a convention line the way `name`/`description`/`type` are, which would
make the service second-guess ai-os's own classification instead of just recording it.

### D8. `scope` defaults to `""`, accepted under either `scope` or `repo_name`; the search filter stays `repo_filter`

`IngestV2Request.scope` uses `AliasChoices("scope", "repo_name")` so existing callers'
vocabulary (`repo_name`) still works while the stored column is named for what it generally
is now (a scope, not necessarily a repo). `SearchV2Request.repo_filter` keeps its v1 name
because it filters on `notes.scope` the same way v1's filter worked, and renaming the filter
too would be change without benefit. **Rejected:** requiring `scope` only, which would break
every v1-vocabulary caller migrating to v2 for no reason — the alias costs nothing and the
default of `""` (not `NULL`) is what keeps the identity unique constraint meaningful, since a
`NULL` would defeat `UNIQUE(client_name, scope, path_or_uri)`.

### D9. `description` joins the FTS document at weight C, not weight B

`description` is note-level and is duplicated onto every passage of its note. At weight B —
the same weight as `content` — a description term would score every passage of a long note
identically, and could outrank a passage that genuinely discusses the term. That is exactly
the "returns a page to read again" failure this redesign exists to remove. `ts_rank_cd`'s
default weights are `{D, C, B, A} = {0.1, 0.2, 0.4, 1.0}`, so weight C contributes real
recall — a note found by its own summary — at half the influence of an actual content hit.
Weight D is left unused and available for a future need. **Rejected:** weight B for
`description`, for the reason above, and weight D, which would make a note's own summary
contribute so little that it would rarely surface a note a caller searched for by its
summary rather than its content.

### D10. `force` bypasses the skip without changing any stored value

A `force` flag on `POST /api/v2/ingest` (and `scripts/reingest.py --force`) re-splits,
re-embeds and rewrites a note's passages even when `content_hash` and `pipeline_version`
both match. It is the escape hatch for the one case the versioned skip key cannot see: a
change in the embedding *model* behind an unchanged backend name and dimension (for
example, a re-pulled Ollama tag). **Rejected:** bumping `pipeline_version` for this case,
which is not possible — the service has no way to detect a model swap that keeps the same
backend name and dimension, so there is nothing for the skip key to compare against. `force`
is a manual override for exactly that blind spot.

### D11. The split threshold decouples "largest passage" from "largest unsplit note"

A body is split at `##`/`###` when it **has** such headings **and** exceeds `TARGET_CHARS`
(1,500 characters). A body with no `##`/`###` headings, or at or below 1,500 characters,
stays one passage — unless it exceeds `MAX_CHARS` (4,000), the hard passage cap, which
triggers paragraph packing regardless of structure. These are deliberately two different
thresholds answering two different questions: `TARGET_CHARS` decides whether structure is
worth using; `MAX_CHARS` is the hard cap on a single passage. **Rejected:** a single 4 KB
"small note" threshold that splits only above it, which was the earlier design and was wrong
by measurement: of the 458 corpus notes at or under 4 KB, 64 carry three or more `##`
headings, and all 64 sit in the 1.5–4 KB band. A single 4 KB threshold would have left every
one of those 64 genuinely structured notes as one multi-topic passage, exactly the retrieval
failure this redesign exists to fix. A headingless 3 KB note, by contrast, has no structure
to exploit and correctly stays whole under the two-threshold rule.

### D12. The skip key is `content_hash` + `pipeline_version`, and `pipeline_version` carries four versions

`pipeline_version()` (`app/core/pipeline.py`) composes the splitter version, the FTS
composition version, the embedding-input composition version, and the embedding backend +
dimension into one comparable string (e.g. `"s2+fts1+ei1+ollama+1024"`). Ingest skips a note
only when **both** the content hash and this string match an already-stored note; a hash
match with a changed pipeline still forces a re-split and re-embed. **Rejected:**
`content_hash` alone, which would silently skip a note whose stored passages were produced
by an older splitter, FTS composition, embedding-input composition, or embedding backend —
exactly the staleness a skip key exists to prevent. Each component is independently
versioned so a change in one (say, the splitter's section-merge rule) forces a re-ingest
without requiring an unrelated bump to, say, the FTS weights.

### D13. `DELETE /api/v2/notes/{id}` exists in slice 1

Erasure is not deferrable: the corpus may contain personal data (ai-os's corpus has a
people/ bucket), and a service that can ingest personal data must be able to remove it from
the first slice that can ingest it at all. Passages go with the note through the existing
`ON DELETE CASCADE` on the composite FK — the route does not delete passages explicitly, so
there is no second place for the tenant filter on that deletion to be wrong. **Rejected:**
deferring `DELETE` to a later slice alongside the vector branch, which would ship an ingest
path with no corresponding erasure path for however long that deferral lasted.

### D14. Every v2 handler sets `@observe(..., capture_input=False, capture_output=False)`

Note bodies are the user's own memory notes and may contain personal data. Langfuse may be
cloud-hosted — `@observe` reads its configuration from the process environment, so a
deployment that sets `LANGFUSE_*` ships whatever is captured off-site. Spans, timings and
errors are still traced on v2; only the payloads are withheld. **Rejected:** v1's default
capture behaviour, which this slice does not change on v1 but does not carry over to v2
either — v2 is a new surface and gets the more conservative default from the start, rather
than being turned on now and revisited later.

## Other decisions worth recording

These are not independently numbered in the plan's fixed list, but they are load-bearing
enough to document.

**Why `tsv` is a written column, not a generated one.** Changing the FTS weights or the
`to_tsvector` configuration later forces a full re-ingest either way — a generated column
recomputed by Postgres on every write cannot see a value that lives on a *different* row
(the note's `name` and `description` are not columns of `passages`). A plain column written
by `NoteRepository.replace_passages` at ingest time is the only way to materialise weight A
(name + heading path) and weight C (description) onto a passage row at all, so there was
never a generated-column option that could express this schema; the choice is recorded here
so a later reviewer does not propose it as a simplification.

**Why `/api/v2/ingest` takes a JSON body instead of form fields.** Starlette enforces a 1 MiB
per-form-field byte limit with no route-level hook to raise it. The corpus's largest known
document (a 112 KB ADR) already approaches that limit in a multi-byte encoding, and nothing
in this service's future (PDFs, larger external sources) gets smaller. A JSON body removes
that ceiling; the service's own `MAX_CONTENT_CHARS` (500,000) becomes the limit that actually
applies, and it is checked explicitly in the handler rather than being an incidental side
effect of the transport.

**The upsert/lock/replace transaction shape, and why the lock is held across the embedding
calls.** One ingest request is one transaction, one commit, in this order: `lock_note`
(`SELECT … FOR UPDATE`, serialising concurrent ingests of the same note identity),
`upsert_note`, embed every passage (calls out to the embedding backend), `replace_passages`,
then `db.commit()`. The row lock is held for the whole span, including the embedding calls,
rather than being released early and re-acquired before the write: releasing it between the
lock and the write would reopen the exact race D3 closes — a second ingest of the same note
could interleave its own lock/upsert/replace between this request's lock and its write,
leaving either an inconsistent split-version mix of passages or a lost update. The cost is
that a slow embedding backend lengthens how long a note's row is locked; that trade is
accepted because correctness under concurrent ingest of the same note matters more than
ingest throughput in this slice, and the lock scope is per-note, not global.

**The tenant-filter-at-top-level rule.** Every `NoteRepository` method puts
`client_name == self.client_name` as a direct argument to its `WHERE` clause — never only
inside a join condition, never only inside an `OR` group, and never only inferred from a
foreign key or an earlier filter in the same request. `replace_passages`'s `DELETE` filters
on `client_name` even though the composite FK already guarantees every passage under a note
belongs to that note's tenant, and `search_fts` filters `client_name` on both `passages` (the
table the GIN index scans) and `notes` (independent of the join condition) rather than
relying on one to imply the other. A filter that is correct only because another constraint
is correct is one migration away from being wrong; this rule is why none of this slice's
tenant isolation depends on that kind of inference.

## Slice boundary

This slice ships: the `notes` + `passages` migration; the pure Markdown splitter
(`app/core/splitter.py`); `POST /api/v2/ingest` (JSON body, versioned skip, `force`,
embeddings and `tsv` written); `GET /api/v2/notes/{id}` and `DELETE /api/v2/notes/{id}`
(both tenant-scoped, 404 with no existence leak); and `POST /api/v2/search`, **FTS only** —
`websearch_to_tsquery` plus `ts_rank_cd`, returning passages with their note's identity,
description and provenance. A re-ingest CLI (`scripts/reingest.py`) and a golden-query
evaluation harness (`scripts/eval_queries.py`) are included so the user can load the real
corpus and measure retrieval quality against a stated gate.

Explicitly **out of scope** for this slice: vector search on the v2 path, rank fusion (RRF),
`hnsw.*` tuning on the v2 path, removing `/api/v1` or `learnings`, context expansion changes,
PDF extraction, and reconciling notes whose source files were deleted. `/api/v1` and
`learnings` are untouched; they are removed only in slice 3, alongside context expansion, as
a major version with no deprecation window. Vector search and RRF are slice 2's work, and
the baseline recorded below is what that work has to beat.

## Retrieval quality baseline

`scripts/eval_queries.py` reports hits@5 against `tests/fixtures/golden_queries.yaml` once
the user has loaded the real ai-os corpus through `scripts/reingest.py` and populated the
golden-query file with the real 15–20 queries (slice 1 ships only 2–3 examples). Fill in
this table after running it:

| date | corpus | notes ingested | passages | queries | hits@5 | gate (15/20) |
|---|---|---|---|---|---|---|
| | | | | | | |

The recorded number here is the **baseline slice 2's RRF work must beat** — slice 2's vector
branch and rank fusion are only justified if they improve on this FTS-only number, not just
on introducing a vector branch for its own sake. A number below the 15/20 gate is a signal to
fix FTS weighting (decision D9) or the splitter's thresholds (decision D11) before adding a
vector branch, not a signal to add the vector branch sooner — slice 2 compounds whatever
retrieval quality slice 1 established; it does not replace it.
