from typing import Any, Sequence

from sqlalchemy import delete, select, text, Row
from sqlalchemy.orm import Session

from app.models.learning import Learning


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
        # cannot leak onto a pooled connection. That scoping relies on the Session
        # having begun a transaction, which SessionLocal(autocommit=False)
        # guarantees by autobegin; with no transaction open, set_config would
        # apply only to its own statement and the search would silently see the
        # defaults again.
        #
        # max_scan_tuples is a safety cap so a pathological query cannot scan the
        # whole table. It is NOT what limits recall in practice: the scan was
        # measured stopping at ~4,786 of 8,010 tuples under every value tried,
        # including 1,000,000, because what ends it is HNSW graph reachability.
        # Raising this will not lengthen a short result set. REINDEX, a higher
        # m/ef_construction, or a partial index that makes client_name an
        # Index Cond: rather than a Filter: would.
        self.db.execute(
            text(
                "SELECT set_config('hnsw.iterative_scan', 'strict_order', true),"
                "       set_config('hnsw.max_scan_tuples', '100000', true)"
            )
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