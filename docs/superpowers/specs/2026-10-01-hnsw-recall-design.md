# Fixing HNSW post-filter starvation in `/api/v1/search`

**Date:** 2026-10-01
**Status:** Approved, pending implementation plan
**Closes:** issue #12

## Context

`POST /api/v1/search` can return an empty result set with HTTP 200 for a client
whose data demonstrably exists. No error, no log, no metric. For an agent-facing
memory API, "I have no memory of that" is indistinguishable from "that was never
stored".

pgvector's HNSW index is **approximate and post-filtered**. The scan walks the
graph, yields candidates in distance order, and `WHERE client_name = ...` is
applied *afterwards* — it appears as `Filter:` in the plan, not `Index Cond:`.
With `hnsw.iterative_scan` at its default `off`, the scan stops once its
candidate window is exhausted, roughly 391 tuples at `ef_search = 40`. If every
candidate in that window belongs to another tenant, the query returns fewer rows
than exist, or none.

Measured against a real container on fully committed, `VACUUM ANALYZE`d data,
25 queries per tenant at `top_k = 10`:

| Tenant's share of table | Rows returned (of 250) | Queries returning zero |
|---|---|---|
| 89% | 250 / 250 | 0 / 25 |
| 10% | 250 / 250 | 0 / 25 |
| 1% | 70 / 250 | 0 / 25 |
| 0.1% | 0 / 250 | 25 / 25 |

Multi-tenancy is not required to trigger it. A single client using `repo_filter`
against a small repository reproduces it, and so does deleted-row churn alone —
20,000 rows inserted, deleted, then 20 live rows inserted returned 0 of 20.
`delete_by_file()` runs on every ingest, so re-ingest churn produces this
whenever autovacuum lags.

The defect predates the stabilization work and was found during its final review.

## The query being fixed

`app/repositories/learning.py`, `LearningRepository.search`:

```python
        stmt = (
            select(Learning, similarity)
            .where(Learning.client_name == self.client_name)
            .order_by(distance)
            .limit(top_k)
        )
```

The `ORDER BY <=>` invites the HNSW index; the `WHERE` is applied after it
produces candidates. That combination is the starvation.

## D1. `strict_order`, not `relaxed_order`

`hnsw.iterative_scan` makes the index keep searching until it has enough rows
surviving the filter. It has two modes, and they promise different things:

- `strict_order` returns rows in exact distance order, so the `top_k` returned
  are genuinely the `top_k` closest. pgvector buffers to achieve this.
- `relaxed_order` is faster but may emit rows slightly out of order. Combined
  with `LIMIT top_k`, a returned row may not be in the true top-k while a
  marginally closer one is cut.

**Chosen: `strict_order`.** The defect being fixed is "search returns the wrong
thing". A subtly wrong result set is the same class of failure, only quieter.
The endpoint also reports a `similarity` score per row, which a caller may
reasonably read as monotonically decreasing.

Rejected: `relaxed_order`, despite being the mode verified during review.
Rejected: exposing the mode as a setting — two code paths behaving differently
makes a future bug report ambiguous about which produced it.

## D2. Settings are applied per transaction, in the repository

```python
self.db.execute(
    text(
        "SELECT set_config('hnsw.iterative_scan', 'strict_order', true),"
        "       set_config('hnsw.ef_search', '200', true),"
        "       set_config('hnsw.max_scan_tuples', '100000', true)"
    )
)
```

`set_config(..., true)` is equivalent to `SET LOCAL` — scoped to the current
transaction and reverted on commit or rollback, so it cannot leak onto a pooled
connection. A single `SELECT` rather than three `SET` statements, because `SET`
accepts one parameter at a time and this runs on every search.

It lives in the repository rather than in engine configuration or `get_db`
because it is a property of *this query shape*, a filtered vector search.
Applied globally it would also affect inserts and deletes, which do not need it.

`SET LOCAL` requires an open transaction. In the request path one is already
open — `get_current_client` queries the database to authenticate before the
endpoint body runs — but the implementation must not depend on that accident,
and SQLAlchemy sessions begin a transaction on first use regardless.

## D3. Recall is improved, not guaranteed

`max_scan_tuples = 100000` bounds the work so a pathological query cannot scan
the entire table. A client holding a truly tiny share of a very large table could
therefore still be starved if 100,000 tuples are not enough to find `top_k`
matches.

This is stated plainly rather than glossed. The alternative — no cap — trades a
silent wrong answer for an unbounded query, which is a worse failure for an API
under a request timeout. `ef_search = 200` is five times the default, which
widens each window before iteration is needed at all.

## D4. Retire the test workaround if the fix permits it

`tests/conftest.py`'s `connection` fixture currently issues
`SET LOCAL enable_indexscan = off`, forcing exact scans so that vector-ordering
assertions are deterministic. It was added because rolled-back rows from earlier
tests crowded out the candidate window — the same starvation this fix addresses,
with a different source of invisible tuples.

Consequently the HNSW path has **no test coverage at all**, which is how #12
stayed invisible: every search test exercises a code path production never takes.

The approach, in order:

1. Apply the fix and add a recall test over a deliberately skewed corpus. It must
   fail before the fix and pass after.
2. Remove the `enable_indexscan = off` line and run the full suite repeatedly.
   At least **fifteen consecutive green runs** before believing the workaround is
   unnecessary.

If flakiness returns, the line is restored and the recall test re-enables index
scans for itself only. That fallback is known to work, so the downside is
bounded, while the upside — every search test exercising the production path — is
what would have caught this defect originally.

## Testing

- **Recall under skew:** seed a bulk client and a small client, search as the
  small client, and assert the full `top_k` is returned. This is the regression
  guard.

  **The corpus sizes are a requirement, not a suggestion, and this spec
  deliberately does not fix them.** The constraint is that the test must
  *reliably* fail before the fix. That is not satisfied by any skew ratio: the
  measurements above show 1% skew returning 70 of 250 rows — partial starvation,
  so a `top_k = 10` query at that ratio might return anything from 0 to 10 and
  the test could pass by luck on the unfixed code. Only at 0.1% was starvation
  total.

  The implementation must therefore choose sizes, demonstrate the test fails on
  the unfixed code across several consecutive runs, and report the sizes used
  along with the observed pre-fix result counts. A sharper ratio is cheap here,
  since the fake embedding function costs almost nothing per row.
- **Ordering still holds:** the existing ranking tests, which use hand-built
  orthogonal vectors so cosine distances are exact, must continue to pass.
  Under `strict_order` they should, and if they do not, that is a finding about
  the mode rather than about the tests.
- **Determinism:** fifteen consecutive full-suite runs after removing the
  exact-scan workaround.

## Cost accepted

`strict_order` buffers results and `ef_search = 200` is five times the default.
Searches get slower in exchange for being correct. At this table's size the
difference is imperceptible; at scale it is a real tradeoff and the right one, as
a fast wrong answer has no value to an agent deciding what it remembers.

## Out of scope

Making an empty result distinguishable from a genuinely empty corpus in logs or
metrics. Worth doing — the issue raises it — but it is observability work rather
than this fix.

The README's "search can silently return fewer results than `top_k`" limitation
is **rewritten, not removed.** D3 is explicit that recall is improved rather than
guaranteed, so deleting the entry outright would replace one false statement with
another. It should describe the remaining bound: iterative scan searches until it
finds `top_k` matches or reaches `max_scan_tuples`, so a client holding a very
small share of a very large table can still receive a short result set.
