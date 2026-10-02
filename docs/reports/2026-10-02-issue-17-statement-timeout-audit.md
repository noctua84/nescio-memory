# QA Audit Report: issue #17 — configurable search `statement_timeout`

**Date:** 2026-10-02
**Auditor:** Pyrrho
**Scope:** `d4eddd6..HEAD` on branch `claude/issue-17-0b5d77`, pinned at
`HEAD_SHA = cc241535bd396295e3f5e2d59a3b20a1e55ce339`
(`BASE_SHA = d4eddd6bd7f93a9591fc70cb2da94fe97893648d`). Two commits
(`f1c4c92` `[impl]`, `cc24153` `[test]`), eight files.
Reviewed against `.sisyphus/plans/issue-17-statement-timeout.md` (rev 2) and
`.sisyphus/discovery/issue-17-statement-timeout.md`.
**Severity Summary:** 0 Critical | 0 Major | 4 Minor | 4 Info

## Executive Summary

The implementation does what the approved rev-2 design asked for, in the place the design
named, and every invariant the brief called out holds: the timeout is genuinely scoped to the
search transaction (ingest, the lifespan schema checks and pooled connections are untouched),
the generic `OperationalError` branch still carries `Retry-After`, no exception text reaches the
client, and the `client_name` tenant filter is unchanged. The full suite is green — 87 passed in
25.7 s with `uv run --locked pytest`.

No finding blocks the merge. The substantive ones are all about the *boundary* of the new
57014 branch rather than its core: the handler cannot tell a search timeout from any other
cancellation, and the headline endpoint test does not in fact exercise the search path — I
confirmed empirically that its fake session raises inside the API-key lookup, so the two
weaknesses are the same weakness seen from two sides.

### Method note

Citations are quoted at `HEAD_SHA`. The "where does the exception actually fire" question was
settled by running the app with the test's own fake session and capturing the raising frame,
not by reading the code (see Finding 1 evidence). The suite was run once, in full.

## Findings

### [MINOR] The new endpoint test never reaches the search query — it exercises the auth lookup

**Category:** Test coverage
**File(s):** `tests/test_error_responses.py:187`, `tests/test_error_responses.py:210-223`
**Confidence:** [VERIFIED]
**Status:** Confirmed (raised as a known item; confirmed, with the mechanism pinned down)

**Description:**
`test_a_canceled_search_statement_returns_503_with_no_retry_after` installs a session whose
*every* `execute` raises:

```python
210:    class _TimingOutSession:
211:        def execute(self, *args, **kwargs):
```

and its leading comment reasons about the search:

```python
187: def test_a_canceled_search_statement_returns_503_with_no_retry_after(
```
```
    # Only LearningRepository.search ever sets a statement_timeout, so a
    # QueryCanceled here is a deterministic "this query was too expensive"
```

But FastAPI resolves `get_current_client` before the route body runs, and
`app/core/security.py:39` is the first `db.execute` in the request:

```python
39:    record = db.execute(stmt).scalar_one_or_none()
```

**Evidence:**
I reproduced the test's wiring in a standalone script (same fake session, same request) and
recorded the stack at the raising call. The fake session was entered exactly once, from
`get_current_client`:

```
Search statement canceled (SQLSTATE 57014): exceeded statement_timeout of 5000 ms
503 {'detail': 'Query exceeded time limit'} {'content-length': '38', 'content-type': 'application/json'}
CALLERS: [['_bootstrap', '_bootstrap_inner', 'run', 'get_current_client', 'execute']]
```

`LearningRepository.search` is never called, so neither the `set_config` at
`app/repositories/learning.py:84` nor the vector `SELECT` at `:102` participates.

**Impact:**
Two things. First, the assertions the test *does* make (status 503, exact detail, absent
`Retry-After`, the 57014 warning) are all properties of the handler, which is path-independent —
so the test is not worthless, it is mis-described. Second, nothing in the suite joins the full
chain: `tests/test_statement_timeout.py` proves the repository sets the timeout and that
Postgres really cancels on it, and this test proves the handler maps `QueryCanceled` correctly,
but no test proves that a *search* that actually exceeds its own timeout produces this response.
A regression that, say, moved the `set_config` after the `SELECT`, or dropped the
`statement_timeout` term from it, would be caught by `test_statement_timeout.py`; a regression in
how the route or the session surfaces the cancellation would not be caught anywhere.

**Reproduction Steps:**
1. Add a counter to `_TimingOutSession.execute` and print the caller.
2. Run the test.
3. Expected (per the name and comment): the second execute, from `LearningRepository.search`.
   Actual: one execute only, from `get_current_client`.

**Recommended Fix:**
Rename to what it tests — e.g. `test_a_canceled_statement_returns_503_with_no_retry_after` — and
reword the comment to describe the handler rather than the search. Separately, if the end-to-end
chain is wanted, add one integration test on the real container: seed an API key via the normal
`client` fixture, `monkeypatch` `settings.statement_timeout_ms` to `1`, and assert the search
endpoint returns 503 with `QUERY_TIMEOUT_DETAIL` and no `Retry-After`. Note the hazard the
existing `test_postgres_actually_cancels_a_statement_over_a_tight_timeout` already documents at
`tests/test_statement_timeout.py:43-48`: a cancellation aborts the transaction, so such a test
must not run on the shared rolled-back `connection` fixture.

---

### [MINOR] The 57014 branch claims a search timeout it cannot verify, and the WARNING asserts a limit that may not have applied

**Category:** Bug (observability / error classification)
**File(s):** `app/core/errors.py:103-109`
**Confidence:** [VERIFIED]
**Status:** Confirmed (raised as a known item; the mapping is a deliberate design choice, the log text is not)

**Description:**
The branch keys on the exception type alone:

```python
103:        if isinstance(exc.orig, psycopg2.errors.QueryCanceled):
104:            logger.warning(
105:                "Search statement canceled (SQLSTATE 57014): exceeded "
106:                "statement_timeout of %d ms",
107:                settings.statement_timeout_ms,
108:            )
```

SQLSTATE 57014 is raised for *any* cancellation of a running statement, not only for one that hit
the timeout this application set: `pg_cancel_backend()` by an operator, a client-side
`conn.cancel()`, and — most realistically — a `statement_timeout` configured by ops at the role,
database or server level (`ALTER ROLE app SET statement_timeout = ...`) all produce it. The
preceding comment states the premise that makes the no-`Retry-After` mapping correct:

```
97:        # Only the search transaction ever sets a statement_timeout (see
98:        # LearningRepository.search), so 57014 here means this particular query
```

That premise is true of *this application's* statements and false of the deployment. The
discovery brief explicitly accepted conflating `pg_cancel_backend` ("acceptable to treat the
same"), so the HTTP mapping is a recorded decision rather than a defect. The log text is not
covered by that decision.

**Evidence:**
My probe above shows the concrete case: the cancellation arose in the API-key lookup, where no
`statement_timeout` had been set — the statement at `app/repositories/learning.py:84`,
`"       set_config('statement_timeout', :statement_timeout, true)"`, had not yet run — and the
handler nonetheless logged `exceeded statement_timeout of 5000 ms`. Both halves of that sentence
("Search statement", "exceeded statement_timeout of 5000 ms") were false for that request.

**Impact:**
The plan designates this WARNING "the only operator signal of HNSW starvation"
(`.sisyphus/plans/issue-17-statement-timeout.md`, Task 1 step 3). An operator who sees it will
conclude that a search exceeded 5000 ms and will go tuning HNSW or raising
`STATEMENT_TIMEOUT_MS` — when the real cause may be an ops-level timeout firing on ingest under
load, or a manual cancel. In the ops-level-timeout case the response is also wrong in the other
direction: that *is* load-dependent, a retry can help, and the client is told not to retry. Low
likelihood, but the failure mode is "operator is actively misdirected by the one signal that
exists", which is worse than silence.

**Reproduction Steps:**
1. With the app running, start a long search and `SELECT pg_cancel_backend(<pid>)` from psql;
   or set `ALTER DATABASE nescio_memory SET statement_timeout = '200ms'` and POST `/api/v1/ingest`
   with a large file.
2. Expected: a log entry that does not assert a cause it cannot know; for the ingest case, a
   retryable 503. Actual: `Search statement canceled (SQLSTATE 57014): exceeded
   statement_timeout of 5000 ms` and a non-retryable "Query exceeded time limit".

**Recommended Fix:**
Keep the mapping (it is the approved decision), and make the log honest about what is known
versus inferred — e.g. `"Statement canceled (SQLSTATE 57014) on %s; the search statement_timeout
is %d ms"` with `request.url.path`, which the handler already receives as `request`. Including
the path costs nothing, distinguishes `/api/v1/search` from `/api/v1/ingest` at a glance, and
leaks nothing (paths are static). Optionally narrow the branch itself to the search path if the
retryability distinction later matters for ingest.

---

### [MINOR] The one operator-facing WARNING is emitted through `logging.lastResort` — the app configures no logging at all

**Category:** Maintainability (observability)
**File(s):** `app/core/errors.py:20`, `app/config.py:34`
**Confidence:** [INFERRED]
**Status:** New

**Description:**
`app/core/errors.py:20` is `logger = logging.getLogger(__name__)` — the first logger call in
`app/`. Nothing in the application configures the `logging` package: there is no
`basicConfig`, no `dictConfig`, no handler attached to the root logger anywhere under `app/`
(grep for `basicConfig|dictConfig` across `app/` returns nothing), and the existing setting
`app/config.py:34` — `    log_level: str = "INFO"` — is declared but read by no code.

**Evidence:**
`app/core/errors.py:20`:
```python
20: logger = logging.getLogger(__name__)
```
`app/config.py:34`:
```python
34:     log_level: str = "INFO"
```
A repository-wide search for a logging configuration call under `app/` finds none; `logging` is
imported only in `app/core/errors.py:11` (`import logging`).

**Impact:**
Under uvicorn, which configures only its own `uvicorn*` loggers and does not attach a handler to
the root logger, `app.core.errors` propagates to a handler-less root and is emitted by Python's
`logging.lastResort` handler: bare text on stderr, no timestamp, no level prefix, no logger name,
not filterable, and ignoring `LOG_LEVEL`. My probe output shows exactly this — the raw line
`Search statement canceled (SQLSTATE 57014): exceeded statement_timeout of 5000 ms` with no
decoration. The warning will be visible, so this is not a silence bug, but the signal the plan
calls load-bearing is the least structured line in the log stream. Marked `[INFERRED]` because I
reasoned from the absent configuration to uvicorn's runtime behaviour rather than observing a
deployed process.

**Reproduction Steps:**
1. `uv run uvicorn app.main:app`, trigger a 57014.
2. Expected: a timestamped, levelled, named record consistent with the rest of the log.
   Actual: an undecorated line from `lastResort`.

**Recommended Fix:**
Out of scope for #17 — but worth a follow-up issue: configure logging once at startup from
`settings.log_level`, which also retires a setting that currently does nothing.

---

### [MINOR] `errors.py` now depends directly on `psycopg2`, against its own stated layering

**Category:** Maintainability
**File(s):** `app/core/errors.py:13`, `app/core/errors.py:92-94`
**Confidence:** [VERIFIED]
**Status:** New

**Description:**
The module docstring and the handler's own comment state the layering rule this import crosses.
`app/core/errors.py:3-6` says the API layer "must not know how a dependency is reached … no
transport detail crosses a layer boundary", and `:92-94` justifies the existing SQLAlchemy
coupling precisely on the grounds that SQLAlchemy *is* the abstraction:

```python
92:        # SQLAlchemy is already the abstraction over the database, so handling its
93:        # exception directly is correct rather than leaky. OperationalError is the
94:        # connectivity family; programming errors are bugs and stay 500s.
```

The new `13: import psycopg2.errors` reaches past that abstraction to the driver.

**Evidence:**
`app/core/errors.py:13`:
```python
13: import psycopg2.errors
```

**Impact:**
Two concrete consequences rather than purity. (1) If the DSN ever moves to psycopg3
(`postgresql+psycopg://`), `exc.orig` becomes a `psycopg.errors.QueryCanceled`, the `isinstance`
silently never matches, and timeouts quietly revert to the old `Database unavailable` +
`Retry-After` behaviour with no test failing — the suite's assertions all go through psycopg2.
`.env.example:14` uses the driver-less `postgresql://` form, which today resolves to psycopg2, so
nothing is broken now. (2) The error-mapping module can no longer be imported without the
driver installed. The plan chose `isinstance` deliberately and for a good reason (a hand-built
`QueryCanceled` has `pgcode is None`), so this is a noted consequence, not a disagreement.

**Reproduction Steps:**
1. Change `DATABASE_URL` to `postgresql+psycopg://…`.
2. Trigger a search timeout.
3. Expected: 503 "Query exceeded time limit", no `Retry-After`. Actual: 503 "Database
   unavailable" with `Retry-After: 30`, and no test fails.

**Recommended Fix:**
Leave as is for #17, and add a one-line comment at the import recording that the branch is
psycopg2-specific and must be revisited on a driver change. A belt-and-braces alternative is
`isinstance(exc.orig, psycopg2.errors.QueryCanceled) or getattr(exc.orig, "pgcode", None) == "57014"`,
which survives a driver swap without weakening the hand-built-exception case.

---

### [INFO] The client-visible `"Query exceeded time limit"` contract is undocumented outside the code

**Category:** Maintainability (documentation)
**File(s):** `README.md:262`, `app/core/errors.py:27`
**Confidence:** [VERIFIED]
**Status:** New

**Description:**
The change adds a second distinguishable 503 shape — same status, different `detail`, and
crucially *no* `Retry-After`, which a well-behaved client's backoff logic will branch on. The
plan required (and the branch delivers) a Configuration-table row for `STATEMENT_TIMEOUT_MS`:

```
262: | `STATEMENT_TIMEOUT_MS` | `5000` … caps the vector-search statement only (not the whole request); must be `> 0`, otherwise the app refuses to start |
```

but nothing documents the response for an API consumer. README has no per-endpoint error table
(the only mention of 503 is the "Current limitations" note at `README.md:353`), and `openapi.json`
documents no error responses, so there is no existing place this *should* have gone.

**Evidence:**
`app/core/errors.py:27`:
```python
27: QUERY_TIMEOUT_DETAIL = "Query exceeded time limit"
```
A search of `README.md` for `503|Retry-After|Database unavailable|time limit` returns one line,
`README.md:353`, which predates this branch.

**Impact:**
Clients cannot learn from the docs that a 503 without `Retry-After` means "do not retry this
query unchanged" — which is the entire behavioural point of the change. No functional impact.

**Recommended Fix:**
A short "Error responses" subsection in README listing the three shapes (503 + `Retry-After`,
503 without it, 500 without it) would serve this and the two pre-existing cases at once. Out of
scope for #17 as planned.

---

### [INFO] Alembic's logger-disabling is a test artifact only — no production path runs Alembic in-process

**Category:** Regression (investigated, refuted)
**File(s):** `alembic/env.py:16-18`, `tests/test_error_responses.py:201`
**Confidence:** [VERIFIED]
**Status:** Confirmed as a non-issue (raised as a known item)

**Description:**
The new test re-enables the handler's logger because the session-scoped `engine` fixture runs
migrations in-process and `alembic/env.py` calls `fileConfig` with the default
`disable_existing_loggers=True`:

```python
17: if config.config_file_name is not None:
18:     fileConfig(config.config_file_name)
```

The question was whether production could suffer the same silencing.

**Evidence:**
Alembic is invoked in exactly two ways in this repository: from the shell
(`README.md:82` `uv run alembic upgrade head`, `README.md:318`) and from
`tests/conftest.py:52` (`from alembic import command`) inside the `engine` fixture. A
repository-wide search excluding `.venv/`, `.git/` and `alembic/` finds no other import of
`alembic` — in particular none in `app/`, which references migrations only as advice in error
text (`app/core/schema_checks.py:43`, `:84`). `app/main.py`'s lifespan runs the two schema
checks and nothing else. There is no Dockerfile, entrypoint script or compose file in the
repository that runs migrations in the API process.

**Impact:**
None in production: the CLI runs in its own process, so `fileConfig` cannot reach the API
process's loggers. The `monkeypatch` at `tests/test_error_responses.py:201` is correctly scoped
to the test and correctly justified in its comment; it does not mask anything. Worth keeping in
mind only if a future deployment ever adds an in-process `command.upgrade()` at startup — that
change would silence the WARNING of Finding 2, and nothing would catch it.

**Reproduction Steps:** n/a — refuted.

**Recommended Fix:** None.

---

### [INFO] `Retry-After` is now expressed two ways in one module

**Category:** Maintainability
**File(s):** `app/core/errors.py:61-62`, `app/core/errors.py:79-81`
**Confidence:** [VERIFIED]
**Status:** New

**Description:**
The new `retry` flag introduces a second spelling of the same conditional-header rule beside the
one the embedding handler already had:

```python
61: def _service_unavailable(detail: str, retry: bool = True) -> JSONResponse:
62:     headers = {"Retry-After": str(RETRY_AFTER_SECONDS)} if retry else None
```
```python
79:        headers = (
80:            {"Retry-After": str(RETRY_AFTER_SECONDS)} if exc.retry_after else None
81:        )
```

**Impact:**
Cosmetic. Two lines to change instead of one if the policy ever moves (for instance to a
`Retry-After` that varies by cause). Noted, not recommended for change inside this PR — the
`retry: bool = True` default is the right shape, since it keeps every existing caller's
behaviour literally unchanged.

**Recommended Fix:** None required.

---

### [INFO] `STATEMENT_TIMEOUT_MS` has a lower bound but no upper bound

**Category:** Security (DoS, residual)
**File(s):** `app/config.py:80-86`
**Confidence:** [VERIFIED]
**Status:** New

**Description:**
The validator rejects `<= 0` as the plan required:

```python
81:     def _statement_timeout_must_be_positive(self) -> "Settings":
82:         if self.statement_timeout_ms <= 0:
```

A misconfiguration in the other direction — `STATEMENT_TIMEOUT_MS=50000000` — passes validation
and reinstates the unbounded-wall-clock hole #17 exists to close, silently.

**Impact:**
Negligible as a live risk: the value is operator-supplied, not attacker-supplied, and an
excessive timeout degrades rather than breaks. Recorded because the failure is silent and the
`.env.example` guidance ("Must exceed the slowest legitimate search") pushes operators upward.

**Recommended Fix:**
Optional. If wanted, a sanity ceiling (say 60000) in the same validator, with the same
fail-at-startup treatment. Not needed for this PR.

## Plan Alignment

| Plan item | Status | Notes |
|-----------|--------|-------|
| Task 1.1 `statement_timeout_ms: int = 5000` + `_statement_timeout_must_be_positive` naming `STATEMENT_TIMEOUT_MS` | Pass | `app/config.py:13`, `:80-86`; message names the env var, matching the `_chunk_window_must_advance` precedent |
| Task 1.2 `set_config('statement_timeout', :timeout, true)` appended to the existing SELECT, bound as a string | Pass | `app/repositories/learning.py:84`, `:86`; bind param, `str(...)`, placed in the existing `set_config` statement |
| Task 1.2 comment extended (transaction scope, ingest/startup unaffected, pooler-safe) | Pass | `app/repositories/learning.py:73-79`; also records why settings is read at call time |
| Task 1.3 `QUERY_TIMEOUT_DETAIL` constant | Pass | `app/core/errors.py:27` |
| Task 1.3 `isinstance(exc.orig, psycopg2.errors.QueryCanceled)` → 503, no `Retry-After` | Pass | `app/core/errors.py:103-109`; `isinstance` as specified, not pgcode |
| Task 1.3 `_service_unavailable()` optional retry flag | Pass | `app/core/errors.py:61-62`; default `True` keeps existing callers identical |
| Task 1.3 WARNING with sqlstate + configured limit, never `str(exc)` | Partial | Emitted and exception-text-free, but asserts a cause it cannot know — Finding 2 |
| Task 1.3 comments (operator signal; why no `Retry-After`) | Pass | `app/core/errors.py:96-102` |
| Task 1.4 `.env.example` + README Configuration row, scope stated | Pass | `.env.example:13-17`, `README.md:262` |
| Task 2 `test_config.py`: default 5000; 0/negative fail with `match="STATEMENT_TIMEOUT_MS"` | Pass | `tests/test_config.py:40-55`; parametrized over `0, -1, -5000`, plus a positive-accepted case |
| Task 2 endpoint test: `QueryCanceled` → 503, exact detail, no `Retry-After`; generic case still has it | Partial | Assertions correct and passing, but it exercises the auth lookup, not the search — Finding 1 |
| Task 2 wiring test: `SHOW statement_timeout` after a real search | Pass | `tests/test_statement_timeout.py:18-31`; distinctive 4321, asserts `"4321ms"` |
| Task 2 driver realism: `OperationalError` with `.orig` a `QueryCanceled`, pgcode 57014 | Pass | `tests/test_statement_timeout.py:34-56`; isolated on its own connection, with the transaction-poisoning hazard explained |
| Task 2 ingest unaffected: fresh transaction shows the server default | Pass | `tests/test_statement_timeout.py:59-67`; `connection` is function-scoped, so "fresh" is real, not order-dependent |
| Acceptance: existing suite green | Pass | 87 passed, 2 warnings, 25.68 s (`uv run --locked pytest`) |
| Acceptance: CI lint clean | Pass (n/a) | The only lint in CI is Spectral over `openapi.json` (`.github/workflows/openapi.yml:79`); no route or model changed, so the committed spec is still in sync |

## Invariant Check Results

| Invariant from the brief | Status | Notes |
|--------------------------|--------|-------|
| (1) Timeout must not affect ingest or startup checks | Pass | `set_config(..., true)` is `SET LOCAL`, scoped to the search transaction. `ingest_file` never calls `search` (`app/api/v1/ingest.py:76-99`), and the lifespan checks (`app/main.py:19-20`) run on the raw engine. `tests/test_statement_timeout.py:59-67` pins the non-leak. |
| (2) `SET LOCAL` relies on an open transaction; `set_config` and the search `SELECT` share one | Pass | `SessionLocal(autocommit=False)` autobegins; FastAPI caches `Depends(get_db)` so `get_current_client` and `search_memory` share one `Session` and therefore one transaction. Proven empirically: `tests/test_statement_timeout.py:30` reads back `"4321ms"` on a *later* statement in the same transaction. |
| (2b) Nothing else in that transaction runs after the search and could be capped unexpectedly | Pass | After `repo.search` the route only maps already-loaded columns to DTOs (`app/api/v1/search.py:29-32`) — no lazy load, no further SQL. `get_db`'s `finally: db.close()` then rolls back. Note the ordering consequence: the auth lookup runs *before* the `set_config`, so it is never capped — which is also why Finding 1's test misses the search. |
| (3) A generic `OperationalError` still returns `Retry-After` | Pass | `app/core/errors.py:110` falls through to `_service_unavailable(DATABASE_UNAVAILABLE_DETAIL)` with the `retry=True` default; `test_database_failure_returns_503` asserts the header and passes. |
| (4) No client-visible leak of internal detail | Pass | Response detail comes from the `QUERY_TIMEOUT_DETAIL` constant; the WARNING formats `settings.statement_timeout_ms` (an int) and a literal, never `str(exc)`. The log is server-side only. |
| (5) Multi-tenant isolation (`client_name`) unchanged | Pass | `app/repositories/learning.py:94` `.where(Learning.client_name == self.client_name)` is byte-identical to `BASE_SHA`; the diff touches only the `set_config` statement and comments above it. |

## Security Assessment (DoS angle from the issue)

The hole #17 names is closed. Before this branch, PR #15's `hnsw.iterative_scan=strict_order`
made a `repo_filter` matching nothing roughly 12× more expensive with nothing bounding wall
clock; the vector `SELECT` at `app/repositories/learning.py:102` is now the only unbounded
statement in the request and it is capped at 5000 ms by default. The other leg of the request,
the embedding call, was already bounded (`app/core/embeddings.py:82`,
`httpx.Client(timeout=OLLAMA_TIMEOUT_SECONDS)`), so the two slow paths are now both finite.

Residual exposure, all pre-existing and all out of scope for this issue:

- **No rate limiting.** A client with a valid key can still issue unbounded concurrent searches;
  the cap converts "one request hangs forever" into "one request burns ≤ 5 s of a backend", which
  is the right trade but not a throttle.
- **Pool queueing is not capped by this change.** `create_engine` at `app/core/db.py:9` takes the
  defaults (pool 5 + 10 overflow, `pool_timeout` 30 s), so under saturation a request can wait
  ~30 s for a connection before its 5 s statement even starts. The README/`.env.example` wording
  ("caps the search statement only, not the whole request") is accurate about this.
- **No `connect_timeout`** — explicitly deferred by the plan's own "Out of scope" section.
- No injection surface is added: the timeout is a bound parameter
  (`{"statement_timeout": str(settings.statement_timeout_ms)}`), not interpolated SQL, and its
  value is an `int` validated at startup.

## Recommendations

Prioritized:

1. **Rename `test_a_canceled_search_statement_returns_503_with_no_retry_after` and fix its
   comment** (Finding 1). One-line change; removes a false claim from the suite.
2. **Make the WARNING say only what it knows**, and include `request.url.path` (Finding 2). This
   is the change with the most operational value in the list — the log is the designated signal
   and it currently misdirects on every non-search cancellation.
3. **Add the end-to-end search-timeout test** (Finding 1, second half), on the real container with
   `statement_timeout_ms` monkeypatched to `1` and its own connection. Optional but it is the one
   genuine coverage gap.
4. **Note the psycopg2 coupling at the import** (Finding 4). A comment, not a refactor.
5. **Follow-up issues, not this PR:** configure application logging from `settings.log_level`
   (Finding 3), document the error-response shapes in README (Finding 5).

Nothing here should hold the merge.

## Files Reviewed

- `app/config.py`
- `app/core/errors.py`
- `app/core/db.py`
- `app/core/security.py`
- `app/core/embeddings.py` (timeout surface only)
- `app/core/schema_checks.py` (references only)
- `app/main.py`
- `app/api/v1/search.py`
- `app/api/v1/ingest.py`
- `app/repositories/learning.py`
- `app/schemas/search.py`
- `tests/conftest.py`
- `tests/test_config.py`
- `tests/test_error_responses.py`
- `tests/test_statement_timeout.py`
- `alembic/env.py`, `alembic.ini`
- `.env.example`, `README.md`
- `.sisyphus/plans/issue-17-statement-timeout.md`, `.sisyphus/discovery/issue-17-statement-timeout.md`
- `.github/workflows/openapi.yml` (lint scope)
