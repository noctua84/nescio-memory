from typing import Any, Sequence

from sqlalchemy import BigInteger, Row, and_, case, cast, delete, or_, select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.models.learning import Learning

# meta["chunk_index"] is a JSONB path expression; .astext renders it as text so
# it can be matched against a regex before the integer cast is attempted. A
# bare cast(..., BigInteger) on a non-numeric or NULL value raises a Postgres
# error (500) rather than failing gracefully, so the CASE/regex guard below
# evaluates the cast only on rows where it is known to succeed. NULL (missing
# key), non-integer values, AND integers outside BigInteger's int8 range all
# fall through to the `else_=None` branch instead of erroring -- the regex
# admits at most 18 digits (plus an optional sign), and 18 nines
# (999999999999999999) is comfortably under int8's max of
# 9223372036854775807 (19 digits), so nothing this regex matches can ever
# overflow the cast below. The cast itself uses BigInteger rather than
# Integer (int4, max ~2.1 billion) because a 19-plus-digit string like
# "99999999999999999999" matches `^-?\d+$` but overflows int4 well before it
# would overflow int8, which previously raised NumericValueOutOfRange (a 500)
# -- exactly the class of error this guard exists to neutralise. A NULL
# chunk_index then makes every bounded comparison in get_chunks_for_ranges
# NULL too, so the malformed row is simply not returned and the affected
# context region degrades to less context rather than to an error.
_CHUNK_INDEX_TEXT = Learning.meta["chunk_index"].astext
_CHUNK_INDEX_IS_INTEGER = _CHUNK_INDEX_TEXT.op("~")(r"^-?\d{1,18}$")
_CHUNK_INDEX = case(
    (_CHUNK_INDEX_IS_INTEGER, cast(_CHUNK_INDEX_TEXT, BigInteger)),
    else_=None,
).label("chunk_index")


class LearningRepository:
    """All pgvector / learnings-table access lives here."""

    def __init__(self, db: Session, client_name: str):
        self.db = db
        self.client_name = client_name

    def add(self, learning: Learning) -> None:
        learning.client_name = self.client_name
        self.db.add(learning)

    def delete_by_file(self, repo_name: str, file_path: str) -> None:
        stmt = delete(Learning).where(
            Learning.client_name == self.client_name,
            Learning.repo_name == repo_name,
            Learning.file_path == file_path,
        )
        self.db.execute(stmt)

    def search(
        self,
        embedding: list[float],
        top_k: int,
        repo_filter: str | None = None,
    ) -> Sequence[Row[Any]]:
        # HNSW is approximate and post-filtered: client_name is applied after the
        # index produces its candidate window, roughly 391 tuples at the default
        # ef_search=40. A client holding a small share of the table could have
        # every candidate filtered away and receive zero rows with no error.
        #
        # iterative_scan makes the index keep searching until it has top_k rows
        # that survive the filter. strict_order rather than relaxed_order: the
        # latter may emit rows out of distance order, which with LIMIT can return
        # a row outside the true top_k while cutting a closer one. Measured recall
        # was identical between the two modes, so the ordering guarantee is free.
        #
        # ef_search is left at its default deliberately. Measured on a 10-of-8010
        # corpus, iterative_scan at ef_search=40 recovered the full top_k on 20 of
        # 20 query vectors -- the same as ef_search=200, at about half the buffers,
        # so production does not pay for a wider first window it does not need.
        #
        # Setting it to 200 did have one real effect, recorded here so it is not
        # rediscovered as a mystery: it widened the window enough to mask candidate
        # window starvation in the TEST suite, whose small corpora compete with
        # earlier tests' rolled-back rows. That is a test concern, and it is handled
        # in tests/conftest.py by forcing exact scans there -- not by carrying a
        # production setting that exists to keep tests green.
        #
        # set_config(..., true) is SET LOCAL, scoped to this transaction, so it
        # cannot leak onto a pooled connection. That scoping relies on an open
        # transaction block; with none, set_config applies only to its own statement
        # and the search would silently see the defaults again. So the setting is
        # read back in a separate statement and a mismatch raises RuntimeError.
        #
        # Session.in_transaction() is not used as a guard because it is False
        # before the first statement (autobegin is lazy) and True under isolation_level
        # AUTOCOMMIT where SET LOCAL still no-ops — it checks the Session, not the
        # database.
        #
        # max_scan_tuples is a safety cap so a pathological query cannot scan the
        # whole table. It is NOT what limits recall in practice: the scan was
        # measured stopping at ~4,786 of 8,010 tuples under every value tried,
        # including 1,000,000, because what ends it is HNSW graph reachability.
        # Raising this will not lengthen a short result set. REINDEX, a higher
        # m/ef_construction, or a partial index that makes client_name an
        # Index Cond: rather than a Filter: would.
        #
        # statement_timeout is set the same way -- scoped to this transaction
        # only, via the same SET LOCAL mechanism -- so ingest commits and the
        # startup schema checks in app/main.py are unaffected; only a search
        # that runs long (e.g. a repo_filter matching nothing, see PR #15) can
        # hit it. Settings is read at call time (attribute access on the
        # module-level `settings` object) rather than captured at import, so
        # tests can monkeypatch it per-case.
        self.db.execute(
            text(
                "SELECT set_config('hnsw.iterative_scan', 'strict_order', true),"
                "       set_config('hnsw.max_scan_tuples', '100000', true),"
                "       set_config('statement_timeout', :statement_timeout, true)"
            ),
            {"statement_timeout": str(settings.statement_timeout_ms)},
        )

        # Read back the HNSW setting to confirm it took effect. If no transaction
        # block is open, SET LOCAL applies only to its own statement and this check
        # will catch the mismatch before the search runs silently with defaults.
        hnsw_setting = self.db.execute(
            text("SELECT current_setting('hnsw.iterative_scan')")
        ).scalar_one()
        if hnsw_setting != 'strict_order':
            raise RuntimeError(
                f"hnsw.iterative_scan setting did not take effect: expected 'strict_order' "
                f"but got {hnsw_setting!r}. set_config(..., true) is SET LOCAL and requires "
                f"an open transaction block to persist to the next statement. With no "
                f"transaction open (e.g., AUTOCOMMIT connection), the setting applies only to "
                f"its own statement and the search would silently run with defaults, "
                f"reinstating #12 (short result sets, HTTP 200)."
            )

        distance = Learning.embedding.cosine_distance(embedding)  # the <=> operator
        similarity = (1 - distance).label("similarity")

        stmt = (
            select(Learning, similarity)
            .where(Learning.client_name == self.client_name)
            .order_by(distance)          # smaller distance = more similar
            .limit(top_k)
        )
        if repo_filter:
            # Filter on the indexed column, not the JSONB field.
            stmt = stmt.where(Learning.repo_name == repo_filter)

        return self.db.execute(stmt).all()

    def get_chunks_for_ranges(
        self, ranges: list[tuple[str, str, int | None, int | None]]
    ) -> Sequence[Row[Any]]:
        """Fetch the chunks covering several index ranges in ONE query.

        Each element of `ranges` is (repo_name, file_path, from_index,
        to_index); a None bound is unbounded on that side, which is how a
        whole-document context region is expressed. Returns (Learning,
        chunk_index) rows ordered by (repo_name, file_path, chunk_index), so
        the caller can group them per document without a second sort.

        One query rather than one per hit: a search returns up to 50 results
        and a per-hit fetch would mean up to 50 round trips on one request
        (see docs/superpowers/specs/2026-10-03-search-context-expansion-design.md,
        D4).

        This sets no statement_timeout of its own, but it is NOT unbounded:
        the only caller runs it inside the same transaction as search(), whose
        `set_config('statement_timeout', ..., true)` is a SET LOCAL and so is
        still in force here. That inheritance is wanted rather than tolerated
        -- a slow context fetch should not be able to hang a search that has
        already found its answer -- and on cancellation the route's blanket
        catch degrades to a result set with no context. Do not "fix" the
        apparent omission by setting one here; do reconsider it if this ever
        gains a caller outside a search transaction, because it would then
        genuinely be uncapped.

        No hnsw.* setting, though: this is a B-tree lookup on the composite
        index (client_name, repo_name, file_path), not a vector search.
        """
        # An empty OR group is not a harmless no-op: or_() with no clauses
        # compiles to a false-ish predicate (and emits a deprecation warning)
        # or, depending on how the disjunction is assembled, to invalid SQL.
        # Either way there is nothing to ask the database, so don't ask it.
        if not ranges:
            return []

        disjuncts = []
        for repo_name, file_path, from_index, to_index in ranges:
            predicates = [
                Learning.repo_name == repo_name,
                Learning.file_path == file_path,
            ]
            # _CHUNK_INDEX evaluates to NULL for a row whose
            # meta["chunk_index"] is missing or not an integer, so a bounded
            # comparison against it is NULL and the row is simply not returned.
            # A malformed row therefore degrades to "no context here" instead
            # of erroring -- the same posture as the guard itself.
            if from_index is not None:
                predicates.append(_CHUNK_INDEX >= from_index)
            if to_index is not None:
                predicates.append(_CHUNK_INDEX <= to_index)
            disjuncts.append(and_(*predicates))

        stmt = (
            select(Learning, _CHUNK_INDEX)
            # The tenant boundary sits at the TOP level of the WHERE clause,
            # ANDed with the whole OR group, deliberately. Repeating
            # client_name inside each disjunct would make correctness depend on
            # every copy being present: one disjunct assembled without it --
            # one `continue` in the loop above, one refactor of the predicate
            # list -- and that branch returns another tenant's rows while the
            # others still look right, so the leak is invisible in review and
            # in any test that does not exercise that exact branch. Here there
            # is one filter, it cannot be partially applied, and no
            # (repo_name, file_path) pair the caller derived from a hit is
            # treated as authorisation to read those rows.
            .where(
                Learning.client_name == self.client_name,
                or_(*disjuncts),
            )
            .order_by(Learning.repo_name, Learning.file_path, _CHUNK_INDEX)
        )
        return self.db.execute(stmt).all()
