from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.config import settings
from app.models.base import Base


class Learning(Base):
    """ Learning model. """
    __tablename__ = "learnings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    repo_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # `metadata` is reserved on DeclarativeBase, so the Python attribute must be named differently
    meta: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False)
    embedding: Mapped[dict] = mapped_column(Vector(settings.embedding_dimension), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now(), onupdate=datetime.now())