from datetime import date, datetime

from sqlalchemy import Date, DateTime, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Note(Base):
    """ One row per ingested unit (a Markdown document). """
    __tablename__ = "notes"

    __table_args__ = (
        # Note identity: a given client cannot ingest the same path twice
        # under the same scope.
        UniqueConstraint(
            "client_name", "scope", "path_or_uri",
            name="uq_notes_client_scope_path",
        ),
        # Exists solely so passages' composite FK can reference
        # (notes.id, notes.client_name) -- Postgres requires the referenced
        # columns of a composite FK to be covered by a unique constraint.
        UniqueConstraint("id", "client_name", name="uq_notes_id_client"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    client_name: Mapped[str] = mapped_column(Text, nullable=False)
    # Holds what v1 calls repo_name. '' rather than NULL because a NULL
    # would defeat the uniqueness of uq_notes_client_scope_path.
    scope: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    path_or_uri: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    type: Mapped[str | None] = mapped_column(Text, nullable=True)
    provenance_kind: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str | None] = mapped_column(Text, nullable=True)
    provenance_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    pipeline_version: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    # Column name "meta", not "metadata" -- that collision is learnings'
    # problem, not ours.
    meta: Mapped[dict] = mapped_column("meta", JSONB, nullable=False, server_default="{}")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
