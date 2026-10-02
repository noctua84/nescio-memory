# QA Audit Report: `fix/hnsw-recall` (whole-branch review before PR closing #12)

> **Read the Outcome addendum at the end of this file before acting on any finding here.** Two of the three Major findings did not survive execution as written: one was adopted and then partly reversed, and one Minor finding that the controller first recorded as unconfirmed turned out to be correct.


**Date:** 2026-10-02
**Auditor:** Pyrrho
**Scope:** `main..fix/hnsw-recall`, pinned at `HEAD_SHA = 3570d94dda5b176229bd1fe746714a6f8a658523`
(base merge-base `802e385fcf1858994d71edaab3e0ac7eecd1bca5`). Seven commits; six files.
Reviewed against `docs/superpowers/specs/2026-10-01-hnsw-recall-design.md` (D1–D4).
**Severity Summary:** 0 Critical | 3 Major | 4 Minor | 3 Info

## Executive Summary

The fix is the right fix, correctly placed, and the transaction scoping is sound — I verified
empirically against pgvector 0.8.6 that `set_config(..., true)` reverts on commit, on rollback and
on rollback-to-savepoint, so nothing leaks onto a pooled connection. Three things need attention
before the PR: the regression test does not pin the query plan it depends on (and I reproduced the
plan flipping away from HNSW on the same data under realistic conditions, in which case the test
passes while exercising nothing); the branch introduces an undeclared hard requirement on
pgvector ≥ 0.8.0 whose violation is a hard `ERROR` on every search; and `ef_search = 200` is
unsupported by the recorded evidence while costing ~2.2× the buffer reads on healthy searches.

The honest decisions in this branch deserve note: the corrected `conftest.py` comment admitting
that D4's stated upside is *not* achieved, and the refusal to delete the README limitation, are both
the right call. Nothing was skipped, `xfail`ed, or marked. The one weakened assertion
(`>= TOP_K - 1`) is defensible as argued, but Finding 7 shows the flake it accommodates is an
artifact of the test's shape rather than of HNSW.

### Method note

Because the load-bearing claims are about planner and index behaviour rather than about Python, I
rebuilt the corpus independently in a `pgvector/pgvector:pg17` container (pgvector 0.8.6): the same
column set, the same two indexes (`ix_learnings_client_repo_path`,
`learnings_embedding_idx` on `hnsw (embedding vector_cosine_ops)`), 8,000 `bulk` rows then 10 `acme`
rows, inserted in that order. Differences from the suite: my rows are committed, autovacuum ran
during the session, and my vectors are uniform-random rather than `fake_embedding`'s normalized
output (cosine distance is scale-invariant, so this should not matter). I did **not** run the pytest
suite — Findings that depend on suite behaviour are tagged accordingly.

---

## Findings

### [MAJOR] The regression test does not pin the query plan, and the plan demonstrably flips away from HNSW

**Category:** Bug (test efficacy)
**File(s):** `tests/test_search_recall.py:58-81`
**Confidence:** [VERIFIED] for the plan flipping; [INFERRED] for the consequence in the suite
**Status:** New
**Blocks the PR:** Yes — this test is the sole guard on #12, and the remedy is one assertion.

**Description:**
The test's whole value rests on the query reaching `learnings_embedding_idx`. Nothing asserts that
it does. If the planner instead chooses `ix_learnings_client_repo_path` + `Sort`, the query becomes
an *exact* search with perfect recall, returns 10 rows, and the test passes green while exercising
neither HNSW nor the fix. That failure mode is silent and indistinguishable from success.

This is not hypothetical. On one fixed corpus and one fixed query vector I observed the plan flip
in both directions under changes that a test suite does not control.

**Evidence:**

The test asserts only a row count and a tenant, at `HEAD_SHA`:

```python
    assert len(rows) >= TOP_K - 1, (
```
(`tests/test_search_recall.py:76`)

```python
    assert {row[0].client_name for row in rows} == {"acme"}
```
(`tests/test_search_recall.py:81`)

**Flip 1 — projection width.** Same 8,010-row corpus, same query vector, no statistics
(`relpages=0, reltuples=-1`). Selecting `learnings.*` + similarity (the production shape, per
`app/repositories/learning.py:61`, `select(Learning, similarity)`) chose HNSW:

```
 Limit  (cost=134.10..514.65 rows=10 width=232)
   ->  Index Scan using learnings_embedding_idx on learnings  (cost=134.10..12692.45 rows=330 width=232)
         Order By: (embedding <=> '<QV>'::vector)
         Filter: (client_name = 'acme'::text)
```

Selecting only `id` from a narrower table chose the exact path:

```
 Limit (actual rows=10 loops=1)
   ->  Sort (actual rows=10 loops=1)
         ->  Index Scan using ix_learnings_client_repo_path on learnings (actual rows=10 loops=1)
               Index Cond: (client_name = 'acme'::text)
```

**Flip 2 — `pg_class` statistics.** Identical data and query, only `pg_class` varied:

| `reltuples` | `relpages` | plan chosen |
|---|---|---|
| `-1` | `0` (fresh container, never vacuumed) | `Index Scan using ix_learnings_client_repo_path` + `Sort` |
| `0` | `2003` (table autovacuumed after rolled-back-only inserts) | `Index Scan using learnings_embedding_idx` |
| `8010` | `2003` (accurate statistics) | `Index Scan using ix_learnings_client_repo_path` + `Sort` |

Two of the three states do **not** use HNSW. The suite's container is session-scoped and long-lived,
and autovacuum mutates `pg_class` during a run (see Finding 9: rolled-back inserts leave dead tuples
that cross the autovacuum threshold while leaving `n_mod_since_analyze` at 0). So which of these
three rows the recall test lands in is decided by autovacuum timing, not by anything the test states.

**Impact:**
The only guard on #12 can become a tautology without any signal. A future change — a column added
to `Learning`, a planner version bump, an `ANALYZE` added to a fixture, autovacuum firing earlier —
makes the test pass for the wrong reason, and the next regression of #12 ships unnoticed.

**Reproduction Steps:**
1. Build the corpus described in the method note.
2. `UPDATE pg_class SET reltuples = 8010, relpages = 2003 WHERE relname = 'learnings';`
3. `EXPLAIN` the production query shape.
4. Expected (for the test to mean anything): `Index Scan using learnings_embedding_idx`.
   Actual: `Index Scan using ix_learnings_client_repo_path` + `Sort` — exact search, 10 rows, test green.

**Recommended Fix:**
Assert the plan inside the test, before the recall assertion. Roughly:

```python
plan = "\n".join(
    r[0] for r in db_session.execute(text("EXPLAIN " + <the same statement>))
)
assert "learnings_embedding_idx" in plan, (
    f"this test only guards #12 while it reaches the HNSW index; planner chose:\n{plan}"
)
```

Driving it through `repo.search` makes getting the identical statement awkward; the cheapest honest
version is to `EXPLAIN` the same SQL text that `LearningRepository.search` builds
(`str(stmt.compile(...))`), or to expose the statement for the test. Either way the test should fail
loudly, not silently, when it stops exercising HNSW. This single assertion also closes Flip 1.

---

### [MAJOR] Undeclared hard requirement on pgvector ≥ 0.8.0; on an older server every search raises `ERROR`

**Category:** Bug / Regression (deployment)
**File(s):** `app/repositories/learning.py:49-55`, `README.md:46`
**Confidence:** [VERIFIED] that `hnsw` is a reserved GUC prefix; [INFERRED] for pgvector < 0.8.0
**Status:** New
**Blocks the PR:** Yes — the minimum version must be stated. A one-line README change suffices.

**Description:**
`hnsw.iterative_scan` and `hnsw.max_scan_tuples` were added in pgvector 0.8.0. PostgreSQL rejects an
unknown parameter under a *reserved* prefix, and the `vector` extension reserves `hnsw`. So on
pgvector 0.7.x the repository's first statement raises an error and `POST /api/v1/search` fails
outright rather than degrading. The branch requires 0.8.0 and says so only in the implementation
plan; `README.md` still states no pgvector minimum at all.

**Evidence:**

The unconditional `set_config` call at `HEAD_SHA`:

```python
        self.db.execute(
            text(
                "SELECT set_config('hnsw.iterative_scan', 'strict_order', true),"
                "       set_config('hnsw.ef_search', '200', true),"
                "       set_config('hnsw.max_scan_tuples', '100000', true)"
            )
        )
```
(`app/repositories/learning.py:49-55`)

The requirement as documented:

```
- **PostgreSQL 16+** with the `pgvector` extension
```
(`README.md:46`)

Probing an unknown `hnsw.*` name on pgvector 0.8.6, in a backend where the library is loaded:

```
 real_default
--------------
 off
ERROR:  invalid configuration parameter name "hnsw.totally_made_up_param"
DETAIL:  "hnsw" is a reserved prefix.
```

In a backend where the library is **not** yet loaded, the same call is accepted silently as a
placeholder (`accepted_unknown_hnsw_guc = 1`). Both halves matter, because the production path hits
the second state first: `get_current_client` queries `api_keys` (no vector value), then the
repository sets the GUCs, and only then does the `<=>` statement load the library.

**Impact:**
On pgvector < 0.8.0, `/api/v1/search` returns a 500 on every request once the backend has touched a
vector value, and silently under-returns (the original #12) before that. The failure appears on the
first search request, not at startup, and no test can see it because the suite pins
`PGVECTOR_IMAGE = "pgvector/pgvector:pg17"` (`tests/conftest.py:69`), which ships 0.8.6.

**Reproduction Steps:**
1. Point the service at a PostgreSQL with pgvector 0.7.x.
2. `POST /api/v1/search` twice on the same pooled connection.
3. Expected: a documented error or degraded recall. Actual: first call under-returns silently,
   subsequent calls fail with `invalid configuration parameter name "hnsw.iterative_scan"`.

**Recommended Fix:**
State the minimum in `README.md:46` (`**PostgreSQL 16+** with `pgvector` **0.8.0 or newer** — the
search sets `hnsw.iterative_scan`, which earlier versions reject`). Better, add a cheap guard: a
migration or startup assertion on
`(SELECT extversion FROM pg_extension WHERE extname = 'vector')`, so the mismatch surfaces at deploy
time with a readable message instead of as a 500 on a user's first search.

---

### [MAJOR] `ef_search = 200` is unsupported by the recorded evidence and costs ~2.2× on healthy searches

**Category:** Performance
**File(s):** `app/repositories/learning.py:52`
**Confidence:** [VERIFIED] for the cost; [VERIFIED] for the evidence gap
**Status:** New
**Blocks the PR:** No — but it is the clearest cost reduction available, and cheap to settle.

**Description:**
Three constants shipped together, and the recorded config matrix cannot separate them. The matrix
tested `off/ef=40`, `strict_order/ef=200`, `strict_order/ef=400`, `relaxed_order/ef=200`,
`relaxed_order/ef=400`. **`strict_order/ef=40` is absent.** So nothing in the evidence shows that
raising `ef_search` 5× contributes anything beyond what `iterative_scan` alone already does — and
`ef_search` is the one constant that charges every search, including the overwhelming majority where
recall was never at risk.

**Evidence:**

The shipped value:

```python
                "       set_config('hnsw.ef_search', '200', true),"
```
(`app/repositories/learning.py:52`)

The common case — searching as `bulk`, which holds 8,000 of 8,010 rows, HNSW forced, same query
vector, 10 rows returned in every case:

| `iterative_scan` | `ef_search` | rows | shared buffer hits | exec time |
|---|---|---|---|---|
| `off` | 40 | 10 | 201 | 0.488 ms |
| `strict_order` | 40 | 10 | 201 | 0.409 ms |
| `strict_order` | 200 | 10 | **450** | **0.675 ms** |

`iterative_scan` alone is free when the first window suffices — identical buffers to `off`.
`ef_search = 200` is what costs: 2.2× the buffer reads and ~1.6× the time, for an identical result.

On the starved side, `ef_search` made no difference to either work or recall in my corpus
(`strict_order/ef=40`: 2,694 buffers, 3.995 ms; `strict_order/ef=200`: 2,684 buffers, 3.771 ms).

**Impact:**
Every healthy search — the great majority of production traffic — pays roughly double the index work
for a benefit that has not been measured. The spec's "Cost accepted" section justifies it as
"widens each window before iteration is needed at all", which is plausible but is exactly the claim
the matrix skipped.

**Recommended Fix:**
Run the missing matrix cell — `strict_order / ef_search = 40` — on the skewed corpus. If recall
matches `ef=200`, drop `set_config('hnsw.ef_search', ...)` entirely and let the server default
stand; the fix then costs nothing on healthy queries. If it does not match, record the number in the
spec, which turns an assumption into a decision.

---

### [MINOR] `max_scan_tuples = 100000` is not the operative bound, and the documented residual-risk mechanism is wrong

**Category:** Maintainability (documentation correctness)
**File(s):** `app/repositories/learning.py:46-48`, `app/repositories/learning.py:53`, `README.md:339-346`
**Confidence:** [VERIFIED]
**Status:** New
**Blocks the PR:** No.

**Description:**
Both the code comment and the README attribute the remaining shortfall to `max_scan_tuples` capping
the work. On my corpus the cap never binds: the iterative scan terminated at exactly the same point
regardless of the cap, the memory multiplier, or `work_mem`. What ends the scan is the HNSW graph's
reachable set — a property no GUC can raise. This matters operationally, because the documentation
points a future operator at a knob that will not help.

**Evidence:**

The claim, at `HEAD_SHA`:

```python
        # cannot leak onto a pooled connection. max_scan_tuples bounds the work,
        # which means recall is much improved but still not guaranteed for a
        # client holding a very small share of a very large table.
```
(`app/repositories/learning.py:46-48`)

```
  that took the result from 1 row to the full 10. But `hnsw.max_scan_tuples` caps the work and HNSW's
```
(`README.md:342`)

Measured, HNSW forced, `acme` (10 of 8,010), one fixed query vector. `Rows Removed by Filter` is the
tuples the scan actually examined:

| `iterative_scan` | `ef_search` | `max_scan_tuples` | `scan_mem_multiplier` | `work_mem` | tuples examined |
|---|---|---|---|---|---|
| `off` | 40 | 20000 | 1 | 4MB | 391 |
| `strict_order` | 200 | 100000 | 1 | 4MB | **4786** |
| `strict_order` | 200 | 100000 | 8 | 4MB | **4786** |
| `strict_order` | 200 | 100000 | 1 | 64MB | **4786** |
| `strict_order` | 200 | **1000000** | 8 | 64MB | **4786** |
| `relaxed_order` | 200 | 100000 | 8 | 64MB | **4786** |

4,786 under every combination, with 8,010 tuples in the table and a cap of up to 1,000,000. The
`off/ef=40` row at 391 matches the spec's "roughly 391 tuples" exactly, which confirms the harness
is measuring the same thing the spec measured.

Two secondary consequences:

- The constant is **untested**. `100000` far exceeds the recall test's 8,010-row corpus, so any value
  above ~8,010 keeps the test green. Changing it to `20000` or `1000000` would not be caught.
- Had memory been the bound, `hnsw.scan_mem_multiplier` (default 1, i.e. `work_mem`) would
  co-determine it, and the spec does not mention that knob. It turned out not to bind here, but it
  deserves a line in the spec alongside `max_scan_tuples` so the next person does not have to
  rediscover it.

**Impact:**
An operator reading the README and seeing short result sets will raise `hnsw.max_scan_tuples`, and
nothing will change. The actual remedies for an unreachable-node tenant are different in kind —
`REINDEX` of `learnings_embedding_idx`, a higher `m`/`ef_construction`, or a per-tenant/partial
index so `client_name` becomes an index condition rather than a `Filter:`.

**Recommended Fix:**
Keep the cap — it is correct as a safety bound and D3's reasoning for having one stands. Correct the
*mechanism* in the comment and the README: recall is bounded by HNSW graph reachability, with
`max_scan_tuples` as an additional work cap that does not normally bind. Note `scan_mem_multiplier`
in the spec. This is a wording fix, not a code change.

---

### [MINOR] Three statements in `test_search_recall.py` were made false by the commit that followed it

**Category:** Maintainability
**File(s):** `tests/test_search_recall.py:8-9`, `tests/test_search_recall.py:61-63`, `tests/test_search_recall.py:65`
**Confidence:** [VERIFIED]
**Status:** New
**Blocks the PR:** No.

**Description:**
Commit `8bc227f` wrote the test while `conftest.py` still forced exact scans; commit `19567ab`
removed that override. `conftest.py`'s own comment was carefully corrected, the test file's was not.
It now documents a fixture behaviour that no longer exists, and carries a statement that does nothing.

**Evidence:**

```python
This test exercises the HNSW path deliberately. The rest of the suite forces
exact scans, which is why this defect stayed invisible.
```
(`tests/test_search_recall.py:8-9` — present tense; the suite no longer forces exact scans)

```python
    Index scans are re-enabled here deliberately: the `connection` fixture
    disables them suite-wide, so without this the HNSW path -- the one production
    uses -- would not be exercised at all.
```
(`tests/test_search_recall.py:61-63` — the `connection` fixture no longer disables them)

```python
    db_session.execute(text("SET LOCAL enable_indexscan = on"))
```
(`tests/test_search_recall.py:65` — `enable_indexscan` defaults to `on`, so this is now a no-op)

Contrast the corrected comment in the same branch:

```python
    # No enable_indexscan override. This fixture used to force exact scans because the
```
(`tests/conftest.py:114`)

**Impact:**
A reader trusting the docstring will believe the suite still forces exact scans, which is the exact
misconception that let #12 hide. The no-op line also invites a reader to think index scans need
re-enabling, obscuring that the plan is now chosen freely — the thing Finding 1 is about.

**Recommended Fix:**
Delete line 65 and the `text` import if it becomes unused; rewrite both comments in the past tense,
and state what is actually true: this test is the only one that reaches HNSW, because it is the only
one with a corpus large enough for the planner to choose it.

---

### [MINOR] The settings silently no-op outside a transaction, and the spec's "must not depend on that accident" is not satisfied

**Category:** Bug (latent)
**File(s):** `app/repositories/learning.py:45-46`, `app/core/db.py:10`
**Confidence:** [VERIFIED] for the no-op; [INFERRED] for the future-regression path
**Status:** New
**Blocks the PR:** No — correct today.

**Description:**
Answering focus item 1 directly: **yes, `set_config(..., true)` is genuinely transaction-scoped
here, and no, these settings cannot leak onto a pooled connection.** I verified all three reversion
paths. But the GUC statement is a *separate round trip* from the search, so correctness depends on a
transaction spanning both. When no transaction is open the setting applies to its own statement only
and the next statement silently sees the default — no error and no warning. Today
`SessionLocal(autocommit=False, ...)` autobegins and the dependency holds; the spec said the
implementation must not *depend* on that, and as written it does.

**Evidence:**

The code's claim:

```python
        # set_config(..., true) is SET LOCAL -- scoped to this transaction, so it
        # cannot leak onto a pooled connection. max_scan_tuples bounds the work,
```
(`app/repositories/learning.py:45-46`)

What keeps it true:

```python
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```
(`app/core/db.py:10`)

No leak — same connection, after `COMMIT`:

```
  inside_txn  | ef_inside
--------------+-----------
 strict_order | 200

 after_commit | ef_after_commit
--------------+-----------------
              |
```

Also reverted after `ROLLBACK` (`after_rollback` empty) and after `ROLLBACK TO SAVEPOINT`
(`after_savepoint_rollback` empty). Calling `search` twice in one request is harmless — the second
call simply re-applies the same values in the same transaction.

The silent degradation, two statements with no explicit transaction:

```
  set_config
--------------
 strict_order

 seen_by_next_statement
------------------------

```

One further verified positive: the GUCs take effect even when applied *before* the `vector` library
is loaded in that backend (`took_effect_before_lib_load = strict_order`,
`after_lib_load = strict_order`), which is the production first-request ordering. So the placeholder
reconciliation is not a problem on 0.8.6.

**Impact:**
Any future move toward autocommit for this read path — `isolation_level="AUTOCOMMIT"`, a raw
`engine.connect()` helper, a read-replica session factory — reinstates #12 invisibly. Nothing fails;
searches just quietly start under-returning again. The recall test cannot catch it, because it runs
through the same `Session` machinery.

**Recommended Fix:**
Cheapest: make the dependency explicit rather than incidental, e.g. assert `self.db.in_transaction()`
before the `set_config` call, or state the requirement in the comment instead of asserting the
(true but less useful) no-leak property. A stronger option is a test that reads
`current_setting('hnsw.iterative_scan')` through the *production* `get_db` session right after a
`search`, which pins the session factory's behaviour rather than the GUC's.

---

### [MINOR] `SMALL_ROWS == TOP_K` forces 100% recall of the client's rows — this is what causes the ~7% flake the assertion was weakened for

**Category:** Maintainability (test design)
**File(s):** `tests/test_search_recall.py:21-23`, `tests/test_search_recall.py:71-76`
**Confidence:** [INFERRED]
**Status:** New
**Blocks the PR:** No. This is my argued challenge to the explicit `>= TOP_K - 1` decision.

**Description:**
Taking up the invitation to challenge: the reasoning for `>= TOP_K - 1` is sound *given the test's
shape*, but the shape is what creates the problem. With `SMALL_ROWS == TOP_K == 10`, the query must
retrieve **every single row the client owns** — the hardest recall task the corpus can pose. It
requires the scan to reach the client's single worst-ranked row, which in a 384-dimensional random
corpus sits near the far end of the distance distribution. A ~7% shortfall there is unremarkable and
is an artifact of asking for exhaustive retrieval, not evidence about HNSW's top-k behaviour.

**Evidence:**

```python
BULK_ROWS = 8_000
SMALL_ROWS = 10
TOP_K = 10
```
(`tests/test_search_recall.py:21-23`)

```python
    # Not `== TOP_K`. Spec D3: recall is improved, not guaranteed -- max_scan_tuples
```
(`tests/test_search_recall.py:71`)

Supporting mechanism, measured: the scan's reachable set on my corpus was 4,786 of 8,010 tuples —
40% of the table was never visited at any setting (Finding 4's table). When the test needs 10 of 10
specific rows, one of them landing outside the reachable set fails the assertion. When it needs the
nearest 10 of a larger pool, that single unreachable row is simply replaced by the next candidate.

**Impact:**
Two costs. First, the assertion had to be loosened to `>= TOP_K - 1`, so a regression to a steady
9 rows now passes silently (detection band: 0–8 fails, 9–10 passes). Second, because
`SMALL_ROWS == TOP_K`, the test cannot check *which* rows came back — the correct answer is
"all of them" — so it is a count test, not a top-k test, and cannot detect a selection or ordering
regression at all. D1 chose `strict_order` specifically to protect ordering, and nothing on this
branch tests that choice under HNSW.

**Recommended Fix:**
Raise `SMALL_ROWS` above `TOP_K` — e.g. `SMALL_ROWS = 40`, `BULK_ROWS = 8_000` (0.5% skew, sharper
than the 1% that produced partial starvation and comfortably above the 0.1% that produced total
starvation). Then `== TOP_K` is likely achievable deterministically, which restores the strict
assertion, and the test gains the ability to assert that the returned similarities are
monotonically non-increasing — which is what D1 actually bought. This needs re-measurement: confirm
the test still fails reliably pre-fix at the new ratio before trusting it. If `== TOP_K` still
flakes at 40 of 8,040, the current assertion is vindicated and that result belongs in the spec.

---

### [INFO] Focus item 4 answered: no test depends on accumulated dead tuples, and rolled-back inserts cannot trigger autoanalyze

**Category:** Bug (investigated, not found)
**File(s):** `tests/conftest.py:105-127`
**Confidence:** [VERIFIED]
**Status:** New
**Blocks the PR:** No.

**Description:**
I checked whether the suite's rolled-back inserts could change planner behaviour for later tests.
Rolled-back inserts do leave dead tuples, but they do **not** advance the autoanalyze counter, so
they cannot give the table statistics mid-run. No test reads or depends on that state.

**Evidence:**
3,000 rows inserted and rolled back, counters reset beforehand:

```
 n_mod_since_analyze | n_dead_tup | n_live_tup | reltuples
---------------------+------------+------------+-----------
                   0 |       3000 |          0 |      8010
```

**Impact:**
The reassuring half: autoanalyze will not silently flip the recall test's plan via `pg_statistic`.
The caveat: `n_dead_tup = 3000` crosses the default autovacuum threshold (`50 + 0.2 × reltuples`),
and an autovacuum **does** update `pg_class.reltuples`/`relpages` — which is the middle row of
Finding 1's table, and the one state in which HNSW *is* chosen. So the 15 consecutive green runs are
consistent with the fix working and also consistent with autovacuum timing happening to be
favourable. Finding 1's plan assertion is what distinguishes them; I would not try to settle it with
more runs.

---

### [INFO] No `statement_timeout`, while the worst-case search cost rose ~12×

**Category:** Performance
**File(s):** `app/core/db.py:9`
**Confidence:** [VERIFIED] for the absence and the measurement
**Status:** New
**Blocks the PR:** No — follow-up.

**Description:**
`grep` for `statement_timeout`, `connect_args`, `isolation_level` and `pool_size` across `app/` and
`README.md` returns nothing; the engine is `create_engine(settings.database_url, pool_pre_ping=True)`
(`app/core/db.py:9`). Meanwhile the per-search ceiling went from ~391 tuples to 4,786 measured (and
to `max_scan_tuples` by design), and the most expensive case is now the *cheapest to request*: a
client with no matching rows. `off/40` on a non-existent client took 0.669 ms; `strict_order/200`
took 3.358 ms with 2,684 buffer hits versus 345.

**Impact:**
Any authenticated client can issue the system's most expensive query repeatedly by passing a
`repo_filter` that matches nothing. At the current table size this is microseconds and irrelevant.
It scales with the table, and the spec's "at scale it is a real tradeoff" is the right read — a
`statement_timeout` is the standard bound and costs one line in `connect_args`.

---

### [INFO] Spec conformance, D1–D4

**Confidence:** [VERIFIED]

- **D1 `strict_order`:** honoured. `grep -n "relaxed_order" app/` returns nothing; the rejected mode
  is absent from production code. The rationale is sound and my measurements support the "costs
  nothing" claim — `strict_order` and `relaxed_order` examined an identical 4,786 tuples at matched
  settings (Finding 4's table). I see no reason to revisit D1. One gap: no test asserts the ordering
  guarantee D1 was chosen for, under HNSW (see Finding 7).
- **D2 per-transaction, in the repository:** honoured and correct. Verified no leak; see Finding 6
  for the one unsatisfied clause.
- **D3 improved, not guaranteed:** honoured in spirit — the README was rewritten rather than deleted,
  which is the right call. The stated *mechanism* is wrong; see Finding 4.
- **D4 retire the workaround:** honoured, and handled unusually well. The branch removed the
  override, ran the 15 runs, and then **corrected the comment to admit D4's stated upside was not
  achieved** (`tests/conftest.py:121-124`: "This does NOT mean every search test now exercises the
  HNSW path"). Reporting that the justification did not pan out, rather than quietly keeping the
  change, is the right behaviour and I want it on the record.

---

## Regression Check Results

| Feature/Contract | Status | Notes |
|---|---|---|
| `search` signature unchanged | Pass | `app/repositories/learning.py:28-33` identical to base; `app/api/v1/search.py` untouched |
| No settings leak to pooled connections | Pass | Verified: reverts on COMMIT, ROLLBACK and ROLLBACK TO SAVEPOINT |
| `search` called twice in one request | Pass | Idempotent re-application within the same transaction |
| GUCs effective before `vector` library load | Pass | Verified on 0.8.6: placeholder reconciliation preserves the values |
| D1: `relaxed_order` absent from `app/` | Pass | `grep` clean |
| No bare `SET` in the repository | Pass | Only `set_config(..., true)` |
| Ordering contract (`similarity` monotonic) | Warning | Preserved by `strict_order`, but untested on the HNSW path (Finding 7) |
| Regression test can fail if fix is reverted | Warning | Yes *while* it reaches HNSW; nothing pins that (Finding 1) |
| `BULK_ROWS = 8_000` doing real work | Pass | It is what makes the planner choose HNSW at all; 4,000 reportedly did not reproduce |
| Query vector is meaningful | Warning | Adequate for a count test; cannot test selection because `SMALL_ROWS == TOP_K` (Finding 7) |
| Suite determinism after workaround removal | Warning | 15 green runs are consistent with both "fixed" and "favourable autovacuum timing" (Finding 9) |
| Nothing skipped / `xfail`ed / marked | Pass | No `skip` or `xfail` anywhere in `tests/` |
| Chosen constants pinned by a test | Fail | `max_scan_tuples` and `ef_search` can both be changed with the suite staying green |
| pgvector minimum version documented | Fail | Finding 2 |
| OpenAPI drift | Pass | No route signature changed; not independently re-run |

## Recommendations

**Before the PR**
1. Assert the query plan in `tests/test_search_recall.py` (Finding 1). One assertion; converts a
   silent vacuous pass into a loud failure, and closes both plan-flip paths.
2. State the pgvector ≥ 0.8.0 requirement in `README.md:46`, ideally with a startup or migration
   version check (Finding 2).

**Cheap, high value, same pass**
3. Measure `strict_order / ef_search = 40` and drop the `ef_search` override if recall holds
   (Finding 3) — removes ~2.2× buffer overhead from every healthy search.
4. Correct the residual-risk mechanism in the comment and README; note `scan_mem_multiplier`
   (Finding 4).
5. Fix the three stale statements in `tests/test_search_recall.py` and delete the no-op `SET LOCAL`
   (Finding 5).

**Worth doing, can follow**
6. Raise `SMALL_ROWS` above `TOP_K`, re-measure, and restore `== TOP_K` plus an ordering assertion
   (Finding 7).
7. Make the open-transaction dependency explicit (Finding 6).
8. Add a `statement_timeout` (Finding 10).

**Deliberately not filed** (per the review brief): empty-vs-missing observability, the ~18s suite
cost, and the `.superpowers/sdd/task-N-report.md` path convention.

## Files Reviewed

- `app/repositories/learning.py` (full, at `HEAD_SHA`)
- `app/api/v1/search.py`, `app/core/db.py`, `app/models/learning.py` (full)
- `tests/conftest.py`, `tests/test_search_recall.py`, `tests/test_search.py`, `tests/test_smoke.py`,
  `tests/factories.py`, `tests/fakes.py` (full)
- `app/config.py` (`embedding_dimension`), `pyproject.toml`, `.github/workflows/ci.yml` (test invocation)
- `alembic/versions/3d80eb22a17d_composite_learnings_index.py`,
  `alembic/versions/8e6b572b0ae3_create_learnings_table.py`,
  `alembic/versions/58f3ecef1edf_switch_embedding_to_0_6b_dimension.py`
- `README.md` (requirements and Current limitations)
- `docs/superpowers/specs/2026-10-01-hnsw-recall-design.md`,
  `docs/superpowers/plans/2026-10-01-hnsw-recall.md`
- Full `main..HEAD` diff at `-U10` via the supplied review package

Empirical probes were run against a throwaway `pgvector/pgvector:pg17` container (pgvector 0.8.6),
now removed. No file under audit was modified.


---

# Outcome addendum (added by the controller after acting on this report)

Recorded here because the findings above were acted on, and two of them did not
survive contact with the full test suite. Anyone reading a finding in isolation
would be misled without this.

| Finding | Outcome |
|---|---|
| MAJOR — test does not pin the query plan | **Adopted.** An `EXPLAIN` assertion for `learnings_embedding_idx` was added. It has passed in every run since, including runs that failed on other assertions, so it does its job. |
| MAJOR — undeclared pgvector >= 0.8.0 requirement | **Adopted, and the highest-value finding in the report.** Mechanism independently confirmed: an unknown `hnsw.*` parameter raises `InvalidName` once the vector library is loaded. A `verify_pgvector_version` startup check and a README floor were added. |
| MAJOR — `ef_search = 200` is unsupported by the evidence | **Adopted, then partly reversed.** The recall claim was correct: at `ef_search=40`, `iterative_scan` recovered the full `top_k` on 20 of 20 query vectors. The conclusion was wrong. Dropping it made the *full suite* fail 4 of 6 runs, because `ef_search=200` had been masking candidate-window starvation in the suite's small corpora, which is also what the 15 green runs behind the `conftest.py` removal had relied on. Resolution: production keeps the default (cheaper), and `tests/conftest.py` forces exact scans again as a test-only measure — the fallback spec D4 had pre-sanctioned. This report's method note is explicit that the suite was not run, which is exactly where the gap was. |
| MINOR — `max_scan_tuples` is not the operative bound | **Adopted.** Both the code comment and the README were corrected to describe it as a safety cap rather than the limit on recall. |
| MINOR — three statements made false by the following commit | **Adopted**, and then made false a second time by the `conftest.py` reversal above, then corrected again. |
| MINOR — settings no-op outside a transaction | **Documented, not guarded.** The dependency is on `SessionLocal(autocommit=False)` autobegin, which is documented Session behaviour rather than the incidental open transaction spec D2 warned about. Recorded in the comment above the `set_config` call; no production assert added. |
| MINOR — `SMALL_ROWS == TOP_K` forces 100% recall and causes the flake | **Correct, and initially dismissed in error.** The controller recorded it as unconfirmed after running 20 query vectors inside a single container build. The phenomenon is cross-build — each build randomizes the HNSW graph — so that experiment could not have seen it. Re-measured properly: at `SMALL_ROWS = 40`, 12 of 12 fresh builds returned exactly `TOP_K`, correctly ordered, while the unfixed configuration still returns 0-1 rows. The `>= TOP_K - 1` threshold was removed and exact equality restored. |
| INFO — no `statement_timeout` | **Deferred** as a follow-up; not specific to this branch. |
| INFO — dead tuples / autoanalyze | Accepted as analysis; no action needed. |
| INFO — spec conformance D1-D4 | Accepted. D1 was left unchanged on this evidence. |

**Final state:** 78 tests, 15 consecutive green full-suite runs.

**The lesson worth keeping**, which cost two round trips: a measurement's scope
must match the scope of the claim it is testing. A config change validated on the
feature's own corpus still needs a full-suite stability run, and a cross-build
phenomenon cannot be refuted inside one build.
