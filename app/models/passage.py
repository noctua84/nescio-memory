from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKeyConstraint, Index, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.config import settings
from app.models.base import Base


class Passage(Base):
    """ One row per structural split of a note's body. """
    __tablename__ = "passages"

    __table_args__ = (
        # client_name is duplicated here (rather than resolved via a join to
        # notes) so the tenant predicate stays a single-table predicate on
        # the table actually being scanned -- a post-join tenant filter
        # worsens issues #12/#74.
        ForeignKeyConstraint(
            ["note_id", "client_name"],
            ["notes.id", "notes.client_name"],
            ondelete="CASCADE",
            name="fk_passages_note",
        ),
        UniqueConstraint("note_id", "ordinal", name="uq_passages_note_ordinal"),
        Index("ix_passages_client_note", "client_name", "note_id"),
        Index("ix_passages_tsv", "tsv", postgresql_using="gin"),
        # Declared so the model matches the schema the migration creates.
        # It is not created from here -- the migration owns it -- but
        # declaring it keeps autogenerate from reporting it as removable,
        # exactly as learnings_embedding_idx is declared on Learning.
        Index(
            "passages_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    note_id: Mapped[int] = mapped_column(Integer, nullable=False)
    client_name: Mapped[str] = mapped_column(Text, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    heading_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dimension), nullable=False
    )
    # A plain column written by ingest -- NOT a generated column.
    tsv: Mapped[str] = mapped_column(TSVECTOR, nullable=False)
