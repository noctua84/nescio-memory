# QA Audit Report: pre-0.5.0 hardening — #48 (context-expansion over-fetch), #42 (zero-norm embeddings), #30 (connect timeout / statement-timeout ceiling)

**Date:** 2026-10-24
**Auditor:** Pyrrho
**Scope:** `origin/main..HEAD` on branch `fix/context-expansion-overfetch`, pinned at
`HEAD_SHA = e71c9bc75ff954c5a5a5fd90558c594ca3150c2e`
(`BASE_SHA = fd46ba3768e5c19cd7e7712d692c162119364eda`). 13 commits, 14 files
(+2555/−64). Reviewed against `.sisyphus/plans/pre-050-hardening.md` and
`.sisyphus/discovery/pre-050-hardening.md` (including its "Updates during execution"
section), and against the brain's repo memory
(`~/.claude/memory/repo/nescio-memory/`: `pgvector-hnsw-is-post-filtered.md`,
`embedding-width-three-way-contract.md`, `sqlalchemy-21-resolves-to-psycopg3.md`).
**Severity Summary:** 0 Critical | 0 Major | 7 Minor | 6 Info

## Executive Summary

All three issues are closed by this branch, in the places the plan named, and the four
load-bearing hazards the brief flagged all hold: the tenant filter sits at the **top level**
of the WHERE clause in *both* context queries (fetch and the new `measure_ranges`), neither
context query sets its own `statement_timeout`, the tail probe survives the new bounds
(verified both by code and by `tests/test_context_bounds.py:693-697`), nothing is capped with
a SQL `LIMIT`, `embedding` is not deferred globally, and a pre-flight failure degrades through
the pre-existing blanket catch to a 200 with no context. The #42 float32 arithmetic is a
genuinely faithful mirror of pgvector's single-precision cosine accumulation, including the
subnormal boundary, and the stored-row guard closes the one hole validation cannot
(`learning.py:230`). Test quality is high: the SQL-text assertion on `embedding`, the
two-tenant measurement test and the seeded property test are the right three tests for this
change.

**No finding blocks the merge.** The substantive ones are all about *claims* rather than
behaviour: the lower-bound proof in `_region_chars_lower_bound` has one unenumerated
counter-example (duplicate `chunk_index` rows, reachable because no unique constraint exists),
one caller-visible note sentence says a document "was not expanded" when it demonstrably was,
and three places in the docs/comments/error message still describe the request as two
statements when it is now three — which matters, because that sentence is the stated rationale
for the new 60 s ceiling.

### Method note

Every citation is quoted at `HEAD_SHA`. The SQL for the NaN predicate and the bind-parameter
type were obtained by compiling the real statement against the psycopg dialect, not read off
the source. The suite was **not** run in this audit (no container started here); the
1265-passed figure is the caller's, carried as `[UNVERIFIED]` where a finding depends on it.
Findings 1, 10 and 11 are reasoned consequences of quoted code and are tagged `[INFERRED]`.

## Findings

### [MINOR] 1. `_region_chars_lower_bound` is not a lower bound when a document has duplicate `chunk_index` rows

**Category:** Bug (correctness of the pre-flight guarantee)
**File(s):** `app/core/context.py:209`, `app/core/context.py:196-203`,
`app/repositories/learning.py:375-382`, `app/models/learning.py:20-37`
**Confidence:** [INFERRED]
**Status:** New

**Description:**
The estimate is

```python
    return min(size.total_chars, max(0, (size.rows - 2) * step))
```
(`app/core/context.py:209`)

and its correctness argument is that `m >= rows - 1`, i.e. at most one measured row (the tail
probe) fails to become part of the region. I verified the join arithmetic it rests on: in
`join_chunk_run` the output length telescopes to `len(last) + (m - 1) * step`
(`app/core/reassembly.py:176-263`), so `(m - 1) * step` is indeed a floor, and the `step` used
here is `min(settings step, min recorded step)` clamped to ≥ 1, which is never larger than the
step `_resolve_window` will pick — so every *window* mismatch errs in the safe direction. I
could not construct a counter-example from chunk sizes, first/last chunk lengths,
`chunk_size`/`chunk_overlap` disagreement between settings and per-row meta, negative indices,
or `document`-mode ranges.

There is one counter-example: **two rows carrying the same `meta["chunk_index"]`**. The
measurement counts rows:

```python
                func.count()
                .filter(_CHUNK_INDEX_IS_INTEGER)
                .label("rows"),
```
(`app/repositories/learning.py:375-377`)

while the builder de-duplicates by index — `content_by_index = {c.chunk_index: c.content ...}`
(`app/core/context.py:618-620`). With every chunk stored twice, `rows` is `2m` and
`total_chars` is `2 ×` the real text, so the "lower bound" is about **twice** the real region
length and the range can be skipped before the fetch even though `build_regions` would have
admitted it. The docstring enumerates the accepted inexact cases — "rows disagree about their
window, or whose index sequence has a hole, or whose overlaps do not verify"
(`app/core/context.py:198-200`) — and duplicate indices are not among them, so the documented
guarantee ("the builder's admitted regions ⊆ the pre-flight's") is overclaimed as written.

Reachability: `learnings` has no unique constraint over `(client_name, repo_name, file_path,
meta->chunk_index)` — `__table_args__` declares only the composite B-tree and the HNSW index
(`app/models/learning.py:20-37`) — so two concurrent ingests of the same path (each doing
`delete_by_file` then insert, `app/api/v1/ingest.py:78`) can interleave into duplicate rows.

**Impact:** A note whose rows were duplicated loses its context expansion (`context_ref: null`
plus a note) even when it fits the budget comfortably. No wrong text, no leak, no error — the
same degradation the feature is designed to fall back to, which is why this is Minor rather
than Major.

**Reproduction Steps:**
1. Seed one note's chunks twice under the same `client_name`/`repo_name`/`file_path` with the
   same `meta.chunk_index` values (bypass the route, or race two ingests).
2. Set `MAX_CONTEXT_CHARS` to ~0.75 × the note's length and search with `context: "document"`.
3. Expected (per the docstring's guarantee): the region is fetched and returned, because the
   builder's real length fits. Actual: `rows` is doubled, the estimate exceeds the budget, the
   range is skipped and every result carries `context_ref: null`.

**Recommended Fix:**
Either make the count index-distinct —
`func.count(func.distinct(_CHUNK_INDEX)).filter(_CHUNK_INDEX_IS_INTEGER)` and, for the chars,
accept `total_chars` as the (now only) cap — or add duplicate indices to the enumerated
inexact cases in both docstrings (`app/core/context.py:196-203` and
`app/core/context.py:302-308`) and soften the README's absolute claim (Finding 5). The
`distinct` variant also makes the estimate right for the far more likely half of the problem
(it is still a lower bound, since de-duplication is exactly what the builder does).

---

### [MINOR] 2. The pre-flight note tells the caller a document "was not expanded" when it *was* expanded

**Category:** Bug (caller-visible contract text)
**File(s):** `app/core/context.py:324`, `app/core/context.py:558-572`,
`tests/test_context_bounds.py:721-726`
**Confidence:** [VERIFIED]
**Status:** New

**Description:**
`admit_ranges` skips individual **ranges** but reports **documents**:

```python
            skipped_documents.add((chunk_range[0], chunk_range[1]))
```
(`app/core/context.py:324`)

and `build_regions` renders that count as a statement about whole documents:

```python
            f"the {budget}-character context budget was reached: "
            f"{bounds.documents_over_budget} document(s) were measured as too "
            f"large to fetch and {omitted} further region(s) were omitted "
```
(`app/core/context.py:559-562`)

In `neighbors` mode one document routinely has several ranges, so a document can have one range
skipped and another fetched and returned. The branch's own test bakes this in: in
`test_both_bounds_firing_quote_the_budget_once` the note asserted is

```python
        "the 5200-character context budget was reached: 1 document(s) were "
        "measured as too large to fetch and 1 further region(s) were omitted "
```
(`tests/test_context_bounds.py:722-724`)

while `test_the_fetch_budget_is_spent_in_rank_order_not_in_planning_order`, the same fixture
and budget, shows `docs/a.md` **present in `contexts`** with region 49–51
(`tests/test_context_bounds.py:678`) — the single "document measured as too large to fetch" is
a document the response expanded.

**Impact:** A caller reading the note concludes a whole note was skipped and may re-query it
with a larger budget, when the truth is "one neighbourhood inside an otherwise expanded note".
The `documents_over_cap` sentence has no such problem (the cap really is per document).

**Reproduction Steps:**
1. Run the branch's `test_both_bounds_firing_quote_the_budget_once` scenario.
2. Inspect `contexts[-1]["note"]` and `contexts[0]["file_path"]`.
3. Expected: the note describes only things absent from `contexts`. Actual: it says one
   document was "measured as too large to fetch" while that document is `contexts[0]`.

**Recommended Fix:**
Count and word the pre-fetch skip in the unit it is actually applied to — e.g. return
`skipped_ranges` and say "`N` region(s) were measured as too large to fetch" — or keep the
document count but only credit documents **none** of whose ranges were admitted. Either way
update the two asserted strings and the README row at `README.md:346`.

---

### [MINOR] 3. The statement-timeout rationale (and the 60 s ceiling it justifies) still counts two statements; a search request now issues three

**Category:** Regression (documentation / operator-facing reasoning)
**File(s):** `app/config.py:164-175`, `README.md:371`, `.env.example:26-27`,
`app/repositories/learning.py:324`
**Confidence:** [VERIFIED]
**Status:** New

**Description:**
The ceiling added for #30 is justified by a per-request statement count:

```python
    # context expansion each get one statement, so 60 seconds per statement is
```
(`app/config.py:166`) and, in the error the operator sees,
```python
                "search and context expansion each get one"
```
(`app/config.py:174`)

The same sentence is in `README.md:371` ("per statement, not per request — search and context
expansion each get one") and `.env.example:26-27`. But #48 added a second context statement
(`measure_ranges`, `app/repositories/learning.py:324`), which inherits the same `SET LOCAL`
(deliberately, and correctly — `app/repositories/learning.py:347-350`). A request can therefore
burn **three** statement timeouts, so at the new 60000 ms ceiling the worst-case wall clock is
~180 s, not ~120 s. The same README row also still opens with "caps the vector-search statement
only (not the whole request)", which contradicts the clause that follows it in the same cell.

**Impact:** An operator sizing `STATEMENT_TIMEOUT_MS` against an upstream gateway timeout
under-estimates the worst case by 50%. No runtime effect.

**Reproduction Steps:**
1. Read `README.md:371` and count the statements a `context: "document"` search executes
   (`search`, `measure_ranges`, `get_chunks_for_ranges`).
2. Expected: the documented count matches the code. Actual: documented 2, actual 3.

**Recommended Fix:**
Say "three statements per request at most (search, the context measurement, the context fetch),
so the worst case is three times this value" in `app/config.py:164-175`, `README.md:371` and
`.env.example:26-27`, and drop "the vector-search statement only" from the README row.

---

### [MINOR] 4. `_statement_timeout_must_be_positive` no longer checks positivity, and its message dropped the consequence of `0`

**Category:** Maintainability (repo's own convention: "error messages explain consequence")
**File(s):** `app/config.py:169-176`
**Confidence:** [VERIFIED]
**Status:** Confirmed (raised as a known nit; confirmed, plus a second half)

**Description:**
```python
    def _statement_timeout_must_be_positive(self) -> "Settings":
        if not 1 <= self.statement_timeout_ms <= 60000:
            raise ValueError(
                "STATEMENT_TIMEOUT_MS must be in range 1..60000, got "
                f"{self.statement_timeout_ms}; per statement, not per request — "
                "search and context expansion each get one"
            )
```
(`app/config.py:169-175`)

Two things: the name now describes a third of what the validator does, and the message traded
the old explanation of *why the floor exists* ("0 disables the timeout in Postgres") for a
sentence about statement counting that is itself inaccurate (Finding 3). The sibling validators
on either side both name their full rule and both state a consequence
(`_max_context_documents_must_be_positive` at `app/config.py:137-145`,
`_db_connect_timeout_must_be_valid` at `app/config.py:152-158`), and the discovery brief
records the convention: "validators are `@model_validator(mode='after')` with a justification
comment above each and an explanatory ValueError message"
(`.sisyphus/discovery/pre-050-hardening.md:22-24`).

**Impact:** An operator who sets `STATEMENT_TIMEOUT_MS=0` now gets a range and a remark about
statement counting, and must go read `app/config.py` to learn that `0` means "no timeout at
all". Pure maintainability/UX.

**Reproduction Steps:**
1. Start with `STATEMENT_TIMEOUT_MS=0`.
2. Expected: the message explains that 0 disables the timeout in Postgres (as it did at
   `BASE_SHA`). Actual: it reports a range and an unrelated clause.

**Recommended Fix:**
Rename to `_statement_timeout_must_be_in_range` and restore both consequences: "0 disables the
timeout in Postgres and negative is rejected by `set_config`; above 60000 would reinstate the
unbounded search of #17". `tests/test_config.py:74` already asserts the range is stated, so it
stays green.

---

### [MINOR] 5. README promises the pre-flight "never" withholds a region that would have fitted

**Category:** Maintainability (doc overclaim)
**File(s):** `README.md:355-359`, `app/core/context.py:196-203`,
`app/core/context.py:302-308`
**Confidence:** [VERIFIED]
**Status:** New

**Description:**
```
to read and admits notes against the same character budget — deliberately underestimating, so a
note that would have fitted is never withheld.
```
(`README.md:357-358`)

The implementation is more honest than this: both docstrings state the approximation
plainly — "the estimate can hold back a document that would in fact have fitted"
(`app/core/context.py:200-201`) and "part of its estimate may belong to a region ranked below
the one being decided" (`app/core/context.py:304-306`) — and Finding 1 adds one more trigger.

**Impact:** The README becomes the thing a future reader trusts over the docstring, and the
next person to debug "why did this note lose its context" starts from a false premise.

**Reproduction Steps:**
1. Compare `README.md:357-358` with `app/core/context.py:196-203`.
2. Expected: the two agree. Actual: the README is unconditional, the code is not.

**Recommended Fix:** "…so a note whose chunks are intact is never withheld; a note with holes
or duplicated chunks may be, and the note on the last region says so."

---

### [MINOR] 6. `context.py`'s module docstring still describes one context query

**Category:** Maintainability (stale comment at the top of the audited module)
**File(s):** `app/core/context.py:5-7`, `app/core/context.py:52-58`
**Confidence:** [VERIFIED]
**Status:** New

**Description:**
```
takes the hits and the already-fetched chunks and returns regions, so the
planning, hole-splitting, window-resolution, budget and mapping logic is all
testable without a database container. The route owns the two I/O steps
(embed + search, then one range fetch) and nothing else.
```
(`app/core/context.py:4-7`)

There are now three I/O steps (embed + search, measure, fetch) and the module has gained a
third responsibility — the two fetch bounds — that the docstring's list does not mention. The
`_expand` docstring was updated for this ("Two queries, or none", `app/api/v1/search.py:44`);
this one was not. Relatedly, `RangeSize`'s docstring says `rows` and `total_chars` "cover only
the rows that could end up in a region" (`app/core/context.py:55-56`), but the tail-probe row
is counted and can never end up in one — which is precisely the row the `- 2` in
`_region_chars_lower_bound` exists to discount.

**Impact:** The first thing a reader of the most intricate module in the change reads is out of
date about how many queries exist and what the module decides.

**Recommended Fix:** Update `app/core/context.py:5-7` to name the three I/O steps and the
bounds phase, and qualify `RangeSize` with "including the tail probe, which cannot".

---

### [MINOR] 7. `literal("NaN", Float)` sends the Python string `'NaN'` and relies on the driver leaving its type for PostgreSQL to infer

**Category:** Maintainability / robustness on the search hot path
**File(s):** `app/repositories/learning.py:236-237`
**Confidence:** [VERIFIED]
**Status:** New

**Description:**
```python
                self_distance != literal("NaN", Float),
                distance != literal("NaN", Float),
```
(`app/repositories/learning.py:236-237`)

Compiled against the psycopg dialect, both render as bind parameters whose **value is the
`str` `'NaN'`** carrying the SQLAlchemy type `FLOAT` (SQLAlchemy's `Float` has no bind
processor, so the string is handed to the driver verbatim):

```
'SELECT learnings.id \nFROM learnings \nWHERE (learnings.embedding <=> learnings.embedding) != %(param_1)s'
{'param_1': 'NaN'}
param_1 'NaN' FLOAT Float
```

This works only because psycopg sends a bare `str` with an unspecified type OID and PostgreSQL
coerces the unknown literal to `float8`. A driver that dumps `str` as `text` turns every search
into `operator does not exist: double precision <> text`. That is not a hypothetical risk in
this repo: the memory note `sqlalchemy-21-resolves-to-psycopg3.md` records that the driver
behind a bare `postgresql://` URL already changed once under a lockfile bump "with no visible
diff", breaking the one CI job that merely imports the app.

**Impact:** A latent, total search outage behind a dependency bump, in exchange for nothing —
the type-correct spellings are the same length.

**Reproduction Steps:**
1. Compile the statement with `literal_binds=True`: it renders `!= NaN` (a bare identifier),
   which is invalid SQL — the construct is only valid via the driver's inference path.
2. Point the URL at a driver that types `str` parameters as `text`.
3. Expected: the filter compares two `float8`s. Actual: no operator, every search 5xx.

**Recommended Fix:** `cast(literal("NaN"), Float)` (explicit `::float8` in the SQL) or
`literal(float("nan"))` (a real float the driver dumps as `'NaN'::float8`). Behaviour is
unchanged; the comparison stops depending on type inference.

---

### [INFO] 8. The extra `embedding <=> embedding` doubles the vector arithmetic per scanned candidate

**Category:** Performance
**File(s):** `app/repositories/learning.py:230`, `app/repositories/learning.py:234-238`
**Confidence:** [INFERRED]
**Status:** New

**Description:** `self_distance = Learning.embedding.cosine_distance(Learning.embedding)`
(`app/repositories/learning.py:230`) is a second full 1024-dimension cosine computed per row
that survives the HNSW scan, in the same `Filter:` position as `client_name`. With
`hnsw.max_scan_tuples = '100000'` (`app/repositories/learning.py:176`) the worst case is 100k
extra 1024-wide dot products per search. The comment claims it "costs nothing extra once
`distance` is already computed for ORDER BY" (`app/repositories/learning.py:225-226`), which is
true of the *query-time* clause but not of the self-distance one. No measurement was taken
either way; the recall suite is the obvious place for one, per
`pgvector-hnsw-is-post-filtered.md`'s rule that "a green streak is evidence about the
configuration it ran under".

**Recommended Fix:** Either measure it on the 8,010-row recall corpus and record the number in
the comment, or (cheaper and exact) add a `CHECK`-style guard at write time plus a one-off
cleanup migration, and keep the read-time filter only as defence. Not for this PR.

---

### [INFO] 9. The pre-flight bounds the transfer, not the database's read of the same text

**Category:** Performance
**File(s):** `app/repositories/learning.py:379-382`
**Confidence:** [INFERRED]
**Status:** New

**Description:** The measurement computes
`func.sum(func.char_length(Learning.content))` (`app/repositories/learning.py:379-381`) over
exactly the rows the fetch would return. `char_length` on a `text` column forces PostgreSQL to
detoast every value, so the server-side I/O and decompression of the over-fetch is still paid
in full; what #48 removes is the wire payload, the psycopg buffer and the Python objects (plus
the embedding, which is the larger win). This is still a real fix — just a narrower one than
"a document that cannot fit is recognised without transferring it"
(`app/api/v1/search.py:54-56`) suggests.

**Recommended Fix:** None required. If server-side cost matters later, `sum(pg_column_size(...))`
on the compressed datum, or `count(*)` plus the recorded window, approximates size without
detoasting.

---

### [INFO] 10. Admitted bytes can exceed `MAX_CONTEXT_CHARS` by the chunk-redundancy factor

**Category:** Performance
**File(s):** `app/core/context.py:205-209`
**Confidence:** [INFERRED]
**Status:** New

**Description:** The estimate prices the *joined region* (`(rows - 2) * step`), while the fetch
transfers the *stored chunks* (`≈ rows * chunk_size`). The ratio is `chunk_size / step`: 1.25×
at the shipped 1000/200, but 10× at a legitimate 1000/900 (`app/config.py` only requires
`0 <= overlap < size`). `total_chars` is the other arm of the `min` but it is the true total, so
it only binds when it is *smaller* than the step-based figure — it cannot pull the estimate down
to the transferred size.

**Recommended Fix:** None for this PR; worth one sentence in the README's bounds paragraph, or a
second arm bounding `total_chars` against a multiple of the budget, if heavy overlap is ever
configured in production.

---

### [INFO] 11. A zero-norm *query* embedding is a client-reachable 503

**Category:** Bug (error classification) — recorded as an accepted product decision
**File(s):** `app/core/embeddings.py:167-172`
**Confidence:** [UNVERIFIED]
**Status:** New

**Description:** `_validated` raises `EmbeddingBackendBadResponse` → 503 + `Retry-After` for a
zero-norm vector (`app/core/embeddings.py:167-172`). For a *stored* row that is unambiguously a
backend fault. For a query it depends on whether any real backend returns a zero vector for an
input a client can send (whitespace-only, punctuation-only, a single emoji). If it does, a
client can drive the service's 5xx rate and `Retry-After` responses at will, and the operator's
alerting cannot distinguish it from a genuinely sick backend. The plan records the 503 as an
explicit user decision (`.sisyphus/plans/pre-050-hardening.md:6`) and the brief lists
"whitespace-only chunks at ingest possibly embedding to zero (unverified)" as out of scope
(`.sisyphus/plans/pre-050-hardening.md:108`), so this is a note, not a reopened decision.

**What would resolve it:** probe both backends (`qwen3-embedding:0.6b`, `all-MiniLM-L6-v2`) with
`" "`, `"."` and `"​"` and record whether any returns a zero-norm vector. If one does, the
query path deserves a 4xx with its own message, separate from the ingest/stored-row fault.

---

### [INFO] 12. Test-coverage gaps

**Category:** Test coverage
**File(s):** `tests/test_context_bounds.py:783-808`, `tests/test_context_bounds.py:832-873`,
`.sisyphus/plans/pre-050-hardening.md:71`
**Confidence:** [VERIFIED]
**Status:** New

Three gaps, none of which I would hold the PR for:

1. **The plan's acceptance item "a failure of the pre-flight degrades exactly like a failed
   context fetch (blanket catch, 200)" (`.sisyphus/plans/pre-050-hardening.md:71`) has no
   test.** `measure_ranges` appears nowhere outside `tests/test_context_bounds.py`
   (grepped across `tests/`). Structurally it is sound — the call is inside the `try` at
   `app/api/v1/search.py:118-119`, and `get_db` only closes the session
   (`app/core/db.py:32-38`), so a poisoned transaction cannot turn into a 500 on teardown — but
   the existing fetch-failure test does not cover the new statement. One `monkeypatch` raising
   inside `measure_ranges` and asserting 200 + `contexts: []` would close it.
2. **The property test re-derives the rank key instead of comparing against `build_regions`.**
   `_reference_rank_key` (`tests/test_context_bounds.py:783-808`) is a line-for-line restatement
   of `_range_rank_key`, which the docstring is candid about; what the safety argument actually
   needs is that `_range_rank_key` agrees with `_rank_key` *as computed from fetched rows*. That
   agreement is covered by exactly one integration test
   (`tests/test_context_bounds.py:619`). I verified the agreement by hand — for undamaged rows
   the range's hit set equals its single region's hit set, and `from_index == piece_lo` — and
   confirmed no hit can sit at `to_index` (a hit there would have merged the spans at
   `app/core/context.py:377`), which is what makes excluding the probe safe rather than merely
   tidy.
3. **`_random_case` never generates `document`-mode ranges (`from_index is None`) or equal
   similarities** (`tests/test_context_bounds.py:845-868`, similarities are
   `round(rng.uniform(0, 1), 6)`), so the two places where `_range_rank_key` substitutes a value
   it cannot know — `0` for an unbounded `from_index`, and tie-breaking — are the two the
   property test does not exercise.

---

### [INFO] 13. Plan deviations (both benign)

**Category:** Plan alignment
**File(s):** `tests/test_context_bounds.py:1`, `app/repositories/learning.py:68-96`
**Confidence:** [VERIFIED]
**Status:** New

- Wave 4/E1 prescribed the #48 tests in `tests/test_search_context.py`
  (`.sisyphus/plans/pre-050-hardening.md:93`); they landed in a new
  `tests/test_context_bounds.py` (921 lines). Every behaviour E1 listed is present — SQL text
  without `embedding` (`:233`), cap keeps rank order (`:385`), beyond-cap nulls + note (`:421`),
  a giant document dropped in document mode (`:490`), generous bounds unchanged (`:734`), tenant
  isolation of **both** queries (`:291`, `:310`, `:322`). The new file is the better call; noted
  only so the deviation is on the record.
- T1 said "do not edit `search()`" and T2 said "NEW method only"; the shared
  `_range_disjuncts` helper (`app/repositories/learning.py:68-96`) does lift the predicate
  loop out of `get_chunks_for_ranges`. The acceptance criterion that mattered holds: the
  `.where(client_name, or_(*disjuncts))` call and the `ORDER BY` are semantically unchanged
  (`app/repositories/learning.py:316-320`), and sharing the predicate is what makes "the rows
  measured are the rows fetched" true by construction rather than by review.

---

### [INFO] 14. On splitting `context.py` (726 lines) — I agree with the decision not to

**Category:** Maintainability
**File(s):** `app/core/context.py`
**Confidence:** [VERIFIED]
**Status:** New

The file carries three phases — planning (`_planned_spans`, `plan_ranges`), bounds
(`documents_within_cap`, `_region_chars_lower_bound`, `admit_ranges`) and building
(`build_regions` and friends) — but they are not three reasons to change: the safety property
of the whole feature is that `_rank_key` (`app/core/context.py:212-224`) and `_planned_spans`
(`app/core/context.py:97-119`) have exactly one definition each, shared across phases. A split
would export both across a module boundary, where the next refactor can give the builder a
private ranking and reintroduce the 9f69c09 defect invisibly. The file is long *and* cohesive,
which is not a defect. The one thing I would change is navigational, not structural: the module
docstring should name the three phases (see Finding 6) so a reader knows there are three before
scrolling.

## Regression Check Results

| Feature/Contract | Status | Notes |
|-------------------|--------|-------|
| Tenant filter at top level of **every** context query | Pass | `learning.py:316-319` (fetch) and `learning.py:390-393` (measure); `_range_disjuncts` deliberately carries neither (`learning.py:76-78`). Covered end to end by `tests/test_context_bounds.py:291,310,322` |
| Context queries set no `statement_timeout` of their own | Pass | Neither method issues `set_config`; both documented as inheriting search's `SET LOCAL` (`learning.py:265-274`, `learning.py:347-350`) |
| Tail probe (`hi + 1`) survives the bounds | Pass | Probe is part of the range, so it is measured and fetched together; `tests/test_context_bounds.py:693-697` asserts `("repo_a","docs/a.md",49,52)` was fetched |
| No SQL `LIMIT` on the context fetch | Pass | `get_chunks_for_ranges` has `where` + `order_by` only (`learning.py:316-321`) |
| `embedding` not deferred globally | Pass | Model unchanged; `search()` still `select(Learning, similarity)` (`learning.py:233`); explicit columns scoped to the one query (`learning.py:298-304`), with the identity-map reason recorded |
| `embedding` absent from both context statements | Pass | Asserted on the SQL text for both modes (`tests/test_context_bounds.py:233-268`) |
| Context failure (incl. pre-flight) degrades to 200, no context | Pass (untested) | Blanket catch covers the new call (`api/v1/search.py:118-138`); `get_db` only closes, so an aborted transaction cannot 500 on teardown (`core/db.py:32-38`). Finding 12.1 |
| Generous bounds → byte-identical output | Pass | `tests/test_context_bounds.py:734`, plus `tests/test_context_bounds.py:444` comparing a capped run against an uncapped one |
| Pre-flight never withholds what the builder would admit | Warning | Holds for undamaged rows (verified by hand and by the seeded property test); Finding 1 (duplicate `chunk_index`) and the documented hole case are exceptions |
| `_range_rank_key` agrees with `build_regions`' `_rank_key` | Pass | Verified by hand for bounded ranges, document ranges, ties and hits outside spans; no hit can occupy `to_index` because `_merge_spans` treats adjacency as overlap (`context.py:377`) |
| `measure_ranges` CASE first-match (ranges disjoint per document) | Pass | `_merge_spans` merges at `lo <= prev_hi + 1`, so unmerged spans satisfy `lo2 >= hi1 + 2 > hi1 + 1 = to_index1`; row sets are disjoint and the `else_=None` group is unreachable under the `or_` WHERE |
| `SearchResult.similarity: float` never null (#42) | Pass | Query side rejected at `embeddings.py:167`; stored side excluded at `learning.py:236`; `tests/test_search.py:395` asserts finiteness for every response |
| Zero-norm detection matches pgvector's float4 arithmetic | Pass | `FLOAT32_MIN_NORMAL = 1.17549435e-38` (`embeddings.py:36`) sits strictly between the largest float32 subnormal (1.1754942106924411e-38) and the smallest normal (1.1754943508222875e-38), so the `<` test is exact for every float32 value; accumulation is rounded per step (`embeddings.py:116-127`), mirroring pgvector's single-precision cosine accumulator |
| #30 connect timeout reaches the engine | Pass | `connect_args={"connect_timeout": ...}` (`core/db.py:17-24`); `tests/test_lazy_engine.py` asserts it; `alembic/env.py` left alone as planned |
| `STATEMENT_TIMEOUT_MS` ceiling is flagged breaking | Pass | `README.md:371` "**breaking in 0.5.0**"; no `!` in any commit subject (`git log origin/main..HEAD`) |
| Log / error hygiene — no customer content | Pass | Blanket catch logs `client_name` + mode only (`api/v1/search.py:131-137`); the #42 message names no component value (`embeddings.py:168-172`); notes and truncation messages name chunk indices and settings numbers only (`context.py:546-578`, `context.py:723-726`) |
| Repo memory: HNSW post-filter contract | Pass | The two NaN clauses are added as post-index `Filter:`s beside `client_name`; `ORDER BY`, `LIMIT`, the three GUCs and the read-back guard are untouched (`learning.py:173-196`) |
| Repo memory: embedding-width three-way contract | Pass | No dimension, backend or column change; `_validated`'s width check is untouched and still ordered ahead of the new norm check (`embeddings.py:131-172`) |

## Recommendations

Prioritised. Nothing here blocks the PR.

1. **Fix before merge (cheap, caller- or operator-visible):** Finding 2 (the note claims a
   document was not expanded when it was — this is API text), Finding 3 (three statements, so
   the ceiling rationale and the worst-case wall clock are wrong in three files), Finding 4
   (validator name + restore the "0 disables the timeout" consequence), Finding 5 (README's
   "never withheld"), Finding 6 (stale module docstring at the top of the module this PR
   rewrites).
2. **Fix before merge or immediately after:** Finding 1 — at minimum add duplicate
   `chunk_index` to the two enumerated inexact-case lists; `count(distinct ...)` is the better
   fix and is two tokens.
3. **Low-risk hardening, this PR or next:** Finding 7 (`cast(literal("NaN"), Float)`),
   Finding 12.1 (one test that a failing `measure_ranges` still returns 200).
4. **Follow-up issues:** Finding 8 (measure the self-distance filter on the recall corpus),
   Finding 11 (probe both backends for client-reachable zero-norm inputs and split the query
   path off the 503 if one exists), Finding 12.3 (extend the property test to document-mode
   ranges and ties), Finding 9/10 (one README sentence each).
5. **Keep as-is:** `context.py` unsplit (Finding 14); the shared `_range_disjuncts`; the new
   test file's location.

**Verdict: ship-with-fixes.** The three issues are genuinely closed, every hazard the brief
named holds, and the two guards for #42 are mutually covering. The fixes in (1) are documentation,
one note sentence and one validator name — an hour of work, no behaviour change — and (2) is the
single place where the implementation's own stated guarantee is stronger than what the code
delivers.

## Files Reviewed

Implementation and docs (at `HEAD_SHA`, with `BASE_SHA` for the diff):
- `app/core/context.py` (full, 726 lines)
- `app/repositories/learning.py` (full)
- `app/api/v1/search.py` (full)
- `app/config.py` (diff + validator block)
- `app/core/db.py` (full)
- `app/core/embeddings.py` (diff + `_validated`)
- `app/core/reassembly.py` (full — to verify the join-length arithmetic the lower bound rests on)
- `app/models/learning.py` (full — constraints, nullability, `meta` attribute name)
- `app/api/v1/ingest.py:78` (the `delete_by_file`-then-insert sequence)
- `README.md` (diff + Configuration table + context section)
- `.env.example` (diff)

Tests:
- `tests/test_context_bounds.py` (full)
- `tests/test_config.py`, `tests/test_error_responses.py`, `tests/test_embeddings.py`,
  `tests/test_search.py`, `tests/test_lazy_engine.py` (diffs and test inventories)
- `tests/test_statement_timeout.py` (grep, for the timeout-inheritance claim)

Context:
- `.sisyphus/plans/pre-050-hardening.md`, `.sisyphus/discovery/pre-050-hardening.md`
- `~/.claude/memory/repo/nescio-memory/`: `MEMORY.md`,
  `pgvector-hnsw-is-post-filtered.md`, `embedding-width-three-way-contract.md`,
  `sqlalchemy-21-resolves-to-psycopg3.md`
- `docs/reports/2026-10-02-issue-17-statement-timeout-audit.md` (style reference)

## Resolution (added after the audit)

All seven Minor findings were fixed on the same branch before merge; Info items 8–11 were filed as follow-ups.

| Finding | Resolution |
|---|---|
| 1 — duplicate `chunk_index` breaks the lower bound | `1a1361a` — `measure_ranges` counts each chunk index once (shorter copy); tests in `fe1abfb` |
| 2 — note says a document "was not expanded" when it was | `30ee8fc` — note and `admit_ranges` count skipped *regions* |
| 3 — "two statements" rationale | `1536109` — config comment, message, README, `.env.example` say up to three |
| 4 — validator name / lost consequence | `1536109` — `_statement_timeout_must_be_in_range`, message explains 0 and >60000 |
| 5 — README "never withheld" overclaim | `1536109` — qualified to well-formed rows |
| 6 — stale `context.py` / `RangeSize` docstrings | `30ee8fc` |
| 7 — `literal("NaN", Float)` binds a string | `ab3a32b` — `cast(literal(float("nan")), Float)` |
| 12.1 — no pre-flight-failure test | `fe1abfb` |
| 8, 9, 10, 11 | filed as #53, #54, #55, #56 |
