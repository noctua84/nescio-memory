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