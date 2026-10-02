# HNSW Recall Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `/api/v1/search` silently returning fewer results than `top_k` — often zero — for clients holding a small share of the `learnings` table.

**Architecture:** `LearningRepository.search` sets three pgvector GUCs per transaction before issuing its query, so the HNSW index keeps searching until it has enough rows surviving the `client_name` filter instead of giving up when its first candidate window is exhausted. A regression test over a deliberately skewed corpus guards it, and the test suite's exact-scan workaround is retired if the fix makes it unnecessary.

**Tech Stack:** pgvector 0.8.6 (`hnsw.iterative_scan` needs ≥ 0.8.0), SQLAlchemy, pytest against the existing testcontainers harness.

**Spec:** `docs/superpowers/specs/2026-10-01-hnsw-recall-design.md` · **Closes:** issue #12

## Global Constraints

- **`strict_order`, never `relaxed_order`.** The latter may emit rows out of distance order, which combined with `LIMIT top_k` can return a row outside the true top-k while cutting a closer one. That is the same class of defect being fixed.
- Exact GUC values: `hnsw.iterative_scan = 'strict_order'`, `hnsw.ef_search = '200'`, `hnsw.max_scan_tuples = '100000'`.
- Settings are applied with `set_config(..., true)` — the `SET LOCAL` equivalent, scoped to the transaction so it cannot leak onto a pooled connection. **Never a bare `SET`.**
- The settings live in `LearningRepository.search`, not in engine configuration, `get_db`, or a migration. They are a property of this query shape.
- Do not change the `search` signature; `app/api/v1/search.py` calls it unchanged.
- Recall is improved, **not guaranteed** — `max_scan_tuples` bounds the work. Do not write code or docs claiming the failure is eliminated.
- Never add `ruff`, `mypy`, or any other checker.
- The suite is at **73 tests** before this plan.
- Commit prefixes: `fix: [fix]` for the production change, `test: [test]` for test-only commits, `docs: [docs]` for the README.

## File Structure

| File | Task | Responsibility |
|---|---|---|
| `tests/test_search_recall.py` | 1, 2 | **new** — the skewed-corpus regression guard |
| `app/repositories/learning.py` | 2 | the GUCs, applied per search |
| `tests/conftest.py` | 3 | retire the exact-scan workaround, if it holds |
| `README.md` | 4 | rewrite the limitation to describe the remaining bound |

---

### Task 1: Prove the defect with a failing test

**Files:**
- Create: `tests/test_search_recall.py`

**Interfaces:**
- Consumes: the `engine` and `db_session` fixtures, `tests.factories.make_learning`, `tests.fakes.fake_embedding`.
- Produces: `tests/test_search_recall.py::test_a_small_client_still_gets_the_full_top_k`, which Task 2 makes pass.

**Why this task is separate:** the spec is explicit that the corpus sizes are a requirement, not a suggestion — the test must *reliably* fail before the fix. Measurement showed 1% skew produces only partial starvation (70 of 250 rows), so a `top_k = 10` query at that ratio could return anywhere from 0 to 10 and pass by luck on unfixed code. Establishing a reliably-failing test first is the only way to know the fix did anything.

- [ ] **Step 1: Write the test**

Create `tests/test_search_recall.py`:

```python
"""Recall under tenant skew.

pgvector's HNSW index is approximate and post-filtered: it yields a bounded
candidate window and `WHERE client_name = ...` is applied afterwards. A client
holding a small share of the table can therefore have every candidate filtered
away and receive an empty result set with HTTP 200.

This test exercises the HNSW path deliberately. The rest of the suite forces
exact scans, which is why this defect stayed invisible.
"""
from sqlalchemy import text

from app.models.learning import Learning
from app.repositories.learning import LearningRepository
from tests.factories import make_learning
from tests.fakes import fake_embedding

# Tuned so starvation is total rather than partial on unfixed code. At 1% skew
# the measured behaviour was 70 of 250 rows -- partial, so a test at that ratio
# could pass by luck. See the plan's Task 1 for the observed figures.
BULK_ROWS = 4_000
SMALL_ROWS = 10
TOP_K = 10


def _seed_skewed_corpus(db_session) -> None:
    # The bulk rows are filler -- only their count and their presence in the
    # index matter -- so they are inserted in one statement rather than through
    # make_learning, which flushes per call. At these volumes that is the
    # difference between one round trip and several thousand.
    db_session.bulk_insert_mappings(
        Learning,
        [
            {
                "client_name": "bulk",
                "repo_name": "bulk_repo",
                "file_path": f"bulk/{index}.md",
                "content": f"bulk chunk {index}",
                "meta": {"chunk_index": index},
                "embedding": fake_embedding(f"bulk chunk {index}"),
            }
            for index in range(BULK_ROWS)
        ],
    )
    # The rows under test go through the factory, so they are built exactly as
    # the rest of the suite builds them.
    for index in range(SMALL_ROWS):
        make_learning(
            db_session,
            "acme",
            repo_name="acme_repo",
            file_path=f"acme/{index}.md",
            content=f"acme chunk {index}",
        )
    db_session.flush()


def test_a_small_client_still_gets_the_full_top_k(db_session):
    """A client holding a fraction of the table must still get top_k rows.

    Index scans are re-enabled here deliberately: the `connection` fixture
    disables them suite-wide, so without this the HNSW path -- the one production
    uses -- would not be exercised at all.
    """
    db_session.execute(text("SET LOCAL enable_indexscan = on"))
    _seed_skewed_corpus(db_session)

    repo = LearningRepository(db_session, client_name="acme")
    rows = repo.search(fake_embedding("anything at all"), top_k=TOP_K)

    assert len(rows) == TOP_K, (
        f"expected {TOP_K} rows for a client with {SMALL_ROWS} of "
        f"{BULK_ROWS + SMALL_ROWS}, got {len(rows)} -- the HNSW candidate window "
        "was exhausted by other clients' rows before top_k matches were found"
    )
    assert {row[0].client_name for row in rows} == {"acme"}
```

- [ ] **Step 2: Confirm it fails, and fails reliably**

Run it **five times** and paste every result:

```bash
for i in 1 2 3 4 5; do uv run pytest tests/test_search_recall.py -q 2>&1 | tail -1; done
```

Expected: **all five fail**, with the assertion reporting fewer than 10 rows.

**If any run passes, the skew is not sharp enough.** Raise `BULK_ROWS` — 8,000, then 16,000 — until all five fail, and report the figures you settled on along with the row counts you observed at each. A test that passes on unfixed code proves nothing, and this step is the only thing standing between a real guard and decoration.

**Also report how long the test takes.** Seeding dominates its runtime, and this test runs on every CI push. If it exceeds roughly ten seconds, say so with the figure — a correct guard that doubles the suite's duration is a trade worth surfacing rather than absorbing silently.

If raising `BULK_ROWS` makes the test unacceptably slow *before* it fails reliably, stop and report that rather than accepting an unreliable test. It would mean this reproduction needs a different shape — for example committing the bulk rows in a session-scoped fixture so the cost is paid once, rather than per test.

- [ ] **Step 3: Commit the failing test**

Committing a known-failing test is deliberate here: it is the evidence the next task is measured against.

```bash
git add tests/test_search_recall.py
git commit -m "test: [test] prove search starves a small client under skew"
```

---

### Task 2: Make the index keep searching

**Files:**
- Modify: `app/repositories/learning.py`

**Interfaces:**
- Consumes: Task 1's failing test.
- Produces: a `search` that returns full `top_k` under skew. Signature unchanged.

- [ ] **Step 1: Add the settings to `search`**

In `app/repositories/learning.py`, add `text` to the existing `sqlalchemy` import:

```python
from sqlalchemy import delete, select, text, Row
```

Then insert the settings at the top of `search`, before the statement is built:

```python
    def search(
        self,
        embedding: list[float],
        top_k: int,
        repo_filter: str | None = None,
    ) -> Sequence[Row[Any]]:
        # Without these, the HNSW index gives up once its first candidate window
        # is exhausted -- roughly 391 tuples at the default ef_search=40. Because
        # client_name is applied as a post-filter rather than an index condition,
        # a client holding a small share of the table can have every candidate
        # filtered away and receive zero rows with no error.
        #
        # iterative_scan makes the index keep searching until it has top_k rows
        # that survive the filter. strict_order rather than relaxed_order: the
        # latter may emit rows out of distance order, which with LIMIT can return
        # a row outside the true top_k while cutting a closer one.
        #
        # set_config(..., true) is SET LOCAL -- scoped to this transaction, so it
        # cannot leak onto a pooled connection. max_scan_tuples bounds the work,
        # which means recall is much improved but still not guaranteed for a
        # client holding a very small share of a very large table.
        self.db.execute(
            text(
                "SELECT set_config('hnsw.iterative_scan', 'strict_order', true),"
                "       set_config('hnsw.ef_search', '200', true),"
                "       set_config('hnsw.max_scan_tuples', '100000', true)"
            )
        )

        distance = Learning.embedding.cosine_distance(embedding)  # the <=> operator
```

Leave the rest of the method, and every other method in the class, exactly as it is.

- [ ] **Step 2: Confirm the recall test now passes, reliably**

```bash
for i in 1 2 3 4 5; do uv run pytest tests/test_search_recall.py -q 2>&1 | tail -1; done
```

Expected: all five pass. Paste every result. One passing run is not enough — the defect was probabilistic, so the fix must be shown to be stable.

- [ ] **Step 3: Confirm ordering still holds**

`strict_order` should preserve exact distance ordering, which the existing ranking tests assert using hand-built orthogonal vectors.

```bash
uv run pytest tests/test_search.py -v
```

Expected: all pass. **If `test_results_are_ordered_by_cosine_distance` fails, stop and report it** — that would be a finding about `strict_order` itself, not about the test, and it would mean the spec's mode choice needs revisiting.

- [ ] **Step 4: Run the whole suite**

```bash
uv run pytest -q
```

Expected: 74 passed.

- [ ] **Step 5: Check for OpenAPI drift**

```bash
uv run python export_openapi.py && git diff --exit-code openapi.json openapi.yaml && echo "no drift"
```

Expected: `no drift` — this changes no route signature. On Windows the export rewrites `openapi.yaml` with CRLF while `.gitattributes` declares LF, so `git status` may show it modified while `git diff` is empty; empty diff means no drift, and regenerated spec files must not be committed.

- [ ] **Step 6: Commit**

```bash
git add app/repositories/learning.py
git commit -m "fix: [fix] make HNSW keep searching past the first candidate window"
```

---

### Task 3: Retire the exact-scan workaround, if the fix permits

**Files:**
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: Task 2's fix.
- Produces: either a suite exercising the real HNSW path, or a documented reason it cannot.

**Why:** `tests/conftest.py:134` issues `SET LOCAL enable_indexscan = off` in the `connection` fixture, forcing exact scans so ranking assertions are deterministic. It exists because rolled-back rows from earlier tests crowded out the candidate window — the same starvation Task 2 addresses, with a different source of invisible tuples. Because of it, every search test exercises a code path production never takes, which is how this defect stayed invisible.

**This task may legitimately end in no change.** That is a real outcome, not a failure.

- [ ] **Step 1: Remove the line**

Delete the `conn.execute(text("SET LOCAL enable_indexscan = off"))` call and its explanatory comment block from the `connection` fixture. Leave `text` imported if anything else uses it; remove the import only if it becomes genuinely unused.

- [ ] **Step 2: Run the full suite fifteen times**

```bash
for i in $(seq 1 15); do uv run pytest -q 2>&1 | tail -1; done
```

Paste all fifteen results.

- **All fifteen pass** → the workaround is unnecessary. Proceed to step 3.
- **Any run fails** → restore the line and its comment exactly, then go to step 4.

Fifteen is not arbitrary: the original flakiness appeared in roughly four runs of five, so a handful of green runs would not distinguish a fix from luck.

- [ ] **Step 3: If all fifteen passed — commit the removal**

Add a brief note in the `connection` fixture explaining why the workaround is gone, so nobody reinstates it:

```python
    # No enable_indexscan override. The ranking tests were previously
    # nondeterministic because the HNSW index gave up once its candidate window
    # was exhausted by earlier tests' rolled-back rows; LearningRepository.search
    # now sets hnsw.iterative_scan, so the index keeps searching and the suite
    # exercises the same path production does.
```

```bash
git add tests/conftest.py
git commit -m "test: [test] exercise the real HNSW path now that recall is fixed"
```

- [ ] **Step 4: If any run failed — restore and report**

Restore the line and comment, confirm `git diff` is empty for `tests/conftest.py`, and **report which test failed, how many of the fifteen, and the assertion text.** Then add the opt-in fallback: the recall test from Task 1 already re-enables index scans for itself, so no further change is needed — but say explicitly in your report that the workaround stays and why, so the spec's D4 is answered either way.

---

### Task 4: Rewrite the README limitation

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 2 and 3.
- Produces: nothing.

**Why:** the Current limitations list says search "can silently return fewer results than `top_k`" with no qualification. That is now wrong in the common case — but deleting it outright would be wrong too, because `max_scan_tuples` bounds the work and recall is improved rather than guaranteed. One false statement must not be replaced with another.

- [ ] **Step 1: Replace the entry**

Find this entry in **Current limitations**:

```
- **Search can silently return fewer results than `top_k`.** pgvector's HNSW index is approximate
  and filters by `client_name` after producing candidates, so a client holding a small share of the
  table can receive an empty result set with HTTP 200 rather than an error.
```

Replace it with:

```
- **Search recall is bounded, not guaranteed.** pgvector's HNSW index is approximate and filters by
  `client_name` after producing candidates. The search now sets `hnsw.iterative_scan` so the index
  keeps looking until it has `top_k` matches, which resolves this for realistic skew — but
  `hnsw.max_scan_tuples` caps the work, so a client holding a very small share of a very large table
  can still receive a short result set rather than an error.
```

Change nothing else in the list, and do not reorder it.

- [ ] **Step 2: Verify**

```bash
grep -n "silently return fewer" README.md || echo "old entry gone"
grep -n "recall is bounded" README.md
```

Expected: the old entry gone, the new one present.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: [docs] describe the bound that remains on search recall"
```

---

## Verification checklist

After Task 4, confirm:

- [ ] `uv run pytest` passes — 74 tests.
- [ ] `tests/test_search_recall.py` passes five consecutive times.
- [ ] `uv run python export_openapi.py && git diff --exit-code openapi.json openapi.yaml` reports no drift.
- [ ] `grep -n "relaxed_order" app/` returns nothing — the rejected mode is absent.
- [ ] `grep -n "SET " app/repositories/learning.py` returns nothing — settings go through `set_config`, not a bare `SET`.
- [ ] `git status --porcelain` is empty.
- [ ] `search`'s signature is unchanged, and `app/api/v1/search.py` is untouched.
- [ ] No `[tool.ruff]` or `[tool.mypy]` was added.
