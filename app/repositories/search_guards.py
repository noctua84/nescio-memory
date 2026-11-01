"""Transaction-local settings every passage search runs under, and the
read-back that proves they took effect.

Split out of passage_search.py, which builds the search statement: these
guards are about the session the statement runs in, not about the statement.
"""

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings

_ITERATIVE_SCAN = "strict_order"
_MAX_SCAN_TUPLES = "100000"


def _apply_search_guards(db: Session) -> None:
    """Set hnsw.iterative_scan, hnsw.max_scan_tuples and statement_timeout for
    this transaction only, then read all three back and raise RuntimeError if
    any did not take effect.

    Called for every search, FTS-only included: the timeout caps every search,
    and the hnsw settings are harmless when no index scan uses them.
    """
    # #12: HNSW is approximate and post-filtered. client_name (and scope) are
    # applied after the index produces its candidate window, roughly 391
    # tuples at the default ef_search=40, so a tenant holding a small share of
    # the table could have every candidate filtered away and get zero rows
    # with a 200. iterative_scan keeps the index searching until enough rows
    # survive the filters. strict_order rather than relaxed_order: the latter
    # may emit rows out of distance order, which under LIMIT can return a row
    # outside the true top n while cutting a closer one; measured recall was
    # identical between the modes, so the ordering guarantee is free.
    #
    # ef_search is left at its default 40 deliberately. 40 is below the
    # vector arm's n (SEARCH_CANDIDATES, default 50), so the first window
    # cannot even hold n rows: iterative scan is REQUIRED for correctness
    # here, not an optimisation. On a 10-of-8010 corpus, iterative_scan at
    # ef_search=40 recovered the full result on 20 of 20 query vectors, the
    # same as ef_search=200 at about half the buffers. A wider ef_search once
    # masked candidate starvation in the test suite; that is handled in
    # tests/conftest.py, not by a production setting that keeps tests green.
    #
    # hnsw.max_scan_tuples is a safety cap so a pathological query cannot
    # scan the whole table. It is NOT a recall lever: the scan was measured
    # stopping at ~4,786 of 8,010 tuples under every value tried, including
    # 1,000,000, because HNSW graph reachability is what ends it. REINDEX,
    # higher m/ef_construction, or a partial index that makes client_name an
    # Index Cond would lengthen a short result set; raising this would not.
    #
    # #15/#17: statement_timeout bounds a search that runs long (e.g. a
    # repo_filter matching nothing). All three use set_config(..., true),
    # i.e. SET LOCAL, scoped to this transaction, so nothing leaks onto a
    # pooled connection and ingest and the startup schema checks are
    # unaffected. `settings` is read at call time so tests can monkeypatch it.
    db.execute(
        text(
            "SELECT set_config('hnsw.iterative_scan', :iterative_scan, true),"
            "       set_config('hnsw.max_scan_tuples', :max_scan_tuples, true),"
            "       set_config('statement_timeout', :statement_timeout, true)"
        ),
        {
            "iterative_scan": _ITERATIVE_SCAN,
            "max_scan_tuples": _MAX_SCAN_TUPLES,
            "statement_timeout": str(settings.statement_timeout_ms),
        },
    )

    # Read back in a SEPARATE statement. The separation is the point: SET
    # LOCAL needs an open transaction block to persist past its own
    # statement, and with none open (e.g. an AUTOCOMMIT connection) each
    # setting covers only the set_config call itself while the search runs
    # uncapped, with default HNSW behaviour, and still answers 200 --
    # reinstating #12 and #15 silently. Reading back inside the same
    # statement would prove nothing.
    #
    # Session.in_transaction() is not the guard: it is False before the first
    # statement (autobegin is lazy) and True under AUTOCOMMIT where SET LOCAL
    # still no-ops. It inspects the Session, not the database.
    #
    # missing_ok=true on the hnsw.* reads: if the vector library has not been
    # loaded into this backend and the placeholder did not survive, a missing
    # setting reads as NULL and is reported below as not having taken effect,
    # rather than surfacing as an unrelated "unrecognized parameter" error.
    iterative_scan, max_scan_tuples, timeout = db.execute(
        text(
            "SELECT current_setting('hnsw.iterative_scan', true),"
            "       current_setting('hnsw.max_scan_tuples', true),"
            "       current_setting('statement_timeout')"
        )
    ).one()

    expected_timeout = _timeout_spellings(settings.statement_timeout_ms)
    for name, shown, ok, expected in (
        ("hnsw.iterative_scan", iterative_scan,
         iterative_scan == _ITERATIVE_SCAN, repr(_ITERATIVE_SCAN)),
        ("hnsw.max_scan_tuples", max_scan_tuples,
         max_scan_tuples == _MAX_SCAN_TUPLES, repr(_MAX_SCAN_TUPLES)),
        ("statement_timeout", timeout,
         timeout in expected_timeout, f"{settings.statement_timeout_ms}ms"),
    ):
        if not ok:
            raise RuntimeError(
                f"{name} setting did not take effect: expected {expected} but "
                f"got {shown!r}. set_config(..., true) is SET LOCAL and "
                f"requires an open transaction block to persist to the next "
                f"statement. With no transaction open (e.g. an AUTOCOMMIT "
                f"connection) it applies only to its own statement and this "
                f"search would run with defaults while still answering 200."
            )


def _timeout_spellings(milliseconds: int) -> frozenset[str]:
    """Every rendering Postgres may give back for this many milliseconds.

    `current_setting('statement_timeout')` normalises an integer GUC to the
    largest unit that divides it evenly -- 5000 comes back as "5s", 60000 as
    "1min", 4321 stays "4321ms", 0 is a bare "0". So the read-back guard cannot
    compare against f"{ms}ms" alone: on the default 5000 that spelling never
    matches and the guard would fire on a correctly applied setting.
    """
    if milliseconds == 0:
        return frozenset({"0", "0ms"})
    spellings = {f"{milliseconds}ms"}
    for unit, size in (("s", 1_000), ("min", 60_000), ("h", 3_600_000)):
        if milliseconds % size == 0:
            spellings.add(f"{milliseconds // size}{unit}")
    return frozenset(spellings)
