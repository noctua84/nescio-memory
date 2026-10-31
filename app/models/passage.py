from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Computed,
    ForeignKeyConstraint,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.config import settings
from app.models.base import Base


class Passage(Base):
    """ One row per structural split of a note's body. """
    __tablename__ = "passages"

    __table_args__ = (
        # client_name and scope are duplicated here (rather than resolved via
        # a join to notes) so the tenant and scope predicates stay
        # single-table predicates on the table actually being scanned -- a
        # post-join filter on HNSW candidates starves results (#12/#74). The
        # composite FK makes Postgres reject a copy that drifts from the
        # note's own values.
        ForeignKeyConstraint(
            ["note_id", "client_name", "scope"],
            ["notes.id", "notes.client_name", "notes.scope"],
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
    # Copied from the note; no server default because ingest always stamps it
    # (NoteRepository.replace_passages) and the composite FK checks it.
    scope: Mapped[str] = mapped_column(Text, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    heading_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dimension), nullable=False
    )
    # A plain column written by ingest -- NOT a generated column.
    tsv: Mapped[str] = mapped_column(TSVECTOR, nullable=False)

    # Computed by Postgres from the STORED vector, never by application code:
    # pgvector's float4 storage can turn a vector Python's own validation
    # accepted into a zero-norm vector once written (#42/#53), and `<=>` on a
    # zero-norm vector is NaN, which would score as a perfect match.
    # SQLAlchemy's Computed() tells the ORM never to try to write this column
    # itself.
    #
    # WIDTH CHANGE WARNING: because this column reads `embedding`, Postgres
    # refuses `ALTER COLUMN embedding TYPE vector(N)` while it exists. A
    # future vector-width migration must DROP this column first, alter
    # embedding's type, then re-add this column with the same generation
    # expression -- see
    # alembic/versions/17be352b1a4c_add_passage_scope_and_degenerate_flag.py.
    embedding_degenerate: Mapped[bool] = mapped_column(
        Boolean,
        Computed("((embedding <=> embedding) = 'NaN'::float8)", persisted=True),
        nullable=False,
    )
