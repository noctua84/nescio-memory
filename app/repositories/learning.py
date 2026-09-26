from typing import Any, Sequence

from sqlalchemy import delete, select, Row
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