"""Passage full-text search (v2). Reads only: ingest and passage writes live in note.py.

Tenant discipline, as stated in app/repositories/note.py's module docstring and
held to here: every statement carries `client_name == self.client_name` as a
DIRECT argument to `.where(...)` -- top level, never only inside a join
condition, never inside an `or_()` group. A filter inside a composed predicate
is correct only while every branch of the composition carries it, and one
refactor can drop it from a single branch while the rest still look right.
At the top level each table gets one predicate that cannot be partially
applied, and no id a caller obtained elsewhere is authorisation to read the
row it names.

search_fts filters both `passages` and `notes`. The join condition pairs them,
and each table also has its own top-level predicate, so either one alone still
enforces the boundary.
"""

from typing import Any, Sequence

from sqlalchemy import Row, and_, func, select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.models.note import Note
from app.models.passage import Passage


class PassageSearchRepository:
    """Passage search over the notes/passages tables. Reads only and commits
    nothing: the route owns the transaction boundary."""

    def __init__(self, db: Session, client_name: str):
        self.db = db
        self.client_name = client_name

    def search_fts(
        self,
        query: str,
        top_k: int,
        repo_filter: str | None = None,
    ) -> Sequence[Row[Any]]:
        """Full-text search over passages, carrying their note's metadata.

        FTS only in this slice: no vector expression and no hnsw.* setting
        anywhere, because nothing here touches the HNSW index.

        statement_timeout is set with set_config(..., true), i.e. SET LOCAL --
        scoped to this transaction, so it cannot leak onto a pooled connection
        or affect ingest and the startup schema checks. `settings` is read at
        CALL time, not captured at import, so tests can monkeypatch it.
        """
        self.db.execute(
            text("SELECT set_config('statement_timeout', :t, true)"),
            {"t": str(settings.statement_timeout_ms)},
        )

        # Read it back in a SEPARATE statement, mirroring
        # app/repositories/learning.py:182-196. The separation is the point:
        # SET LOCAL needs an open transaction block to persist past its own
        # statement, and with none open the timeout covers only the set_config
        # call itself while this search runs uncapped and still answers 200.
        # Reading it back inside the same statement would prove nothing.
        #
        # Session.in_transaction() is not the guard: it is False before the
        # first statement (autobegin is lazy) and True under AUTOCOMMIT where
        # SET LOCAL still no-ops. It inspects the Session, not the database.
        shown = self.db.execute(
            text("SELECT current_setting('statement_timeout')")
        ).scalar_one()
        if shown not in _timeout_spellings(settings.statement_timeout_ms):
            raise RuntimeError(
                f"statement_timeout setting did not take effect: expected "
                f"{settings.statement_timeout_ms}ms but got {shown!r}. "
                f"set_config(..., true) is SET LOCAL and requires an open "
                f"transaction block to persist to the next statement. With no "
                f"transaction open (e.g. an AUTOCOMMIT connection) it applies "
                f"only to its own statement and this search would run uncapped "
                f"while still answering 200."
            )

        q = func.websearch_to_tsquery("english", query)
        # Left on the default weight vector {0.1, 0.2, 0.4, 1.0} deliberately:
        # putting description at weight C depends on those defaults, so a
        # custom vector here would silently rebalance A/B/C.
        rank = func.ts_rank_cd(Passage.tsv, q).label("rank")
        stmt = (
            select(
                Passage.id, Passage.note_id, Passage.ordinal,
                Passage.heading_path, Passage.content, rank,
                Note.name, Note.description, Note.type, Note.path_or_uri,
                Note.scope, Note.provenance_kind, Note.provenance_date,
                Note.status,
            )
            .join(
                Note,
                and_(
                    Note.id == Passage.note_id,
                    Note.client_name == Passage.client_name,
                ),
            )
            # Both client_name predicates are present on purpose: the one on
            # Passage is the single-table predicate on the table the GIN index
            # scans, and the one on Note is independent of the join condition,
            # so dropping either still leaves the boundary enforced.
            .where(
                Passage.client_name == self.client_name,
                Note.client_name == self.client_name,
                Passage.tsv.op("@@")(q),
            )
            # Passage.id is a deterministic tiebreak: ts_rank_cd ties are
            # common and an unstable order makes tests flap.
            .order_by(rank.desc(), Passage.id)
            .limit(top_k)
        )
        if repo_filter:
            stmt = stmt.where(Note.scope == repo_filter)

        return self.db.execute(stmt).all()


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
