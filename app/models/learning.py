from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.config import settings
from app.models.base import Base


class Learning(Base):
    """ Learning model. """
    __tablename__ = "learnings"

    # One composite index rather than three single-column ones. A B-tree serves
    # any prefix of its columns, so this covers client_name alone (every search),
    # client_name with repo_name (a filtered search), and all three
    # (delete_by_file, which runs on every ingest). file_path is never filtered
    # on its own.
    __table_args__ = (
        Index(
            "ix_learnings_client_repo_path",
            "client_name",
            "repo_name",
            "file_path",
        ),
        # Declared so the model matches the schema migration 0001 creates. It is
        # not created from here -- the migration owns it -- but declaring it keeps
        # autogenerate from reporting it as removable, which would otherwise have
        # to be tolerated by an allowlist entry.
        Index(
            "learnings_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    repo_name: Mapped[str] = mapped_column(Text, nullable=False)
    client_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # `metadata` is reserved on DeclarativeBase, so the Python attribute must be named differently
    meta: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dimension), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )