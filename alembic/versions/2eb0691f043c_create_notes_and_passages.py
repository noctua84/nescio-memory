"""create notes and passages

Revision ID: 2eb0691f043c
Revises: 57cfbdfa0d6a
Create Date: 2026-10-29 00:00:00.000000

Adds the v2 storage model alongside the untouched `learnings` table: `notes`
(one row per ingested unit, typed columns) and `passages` (one row per
structural split, with `tsv` and `embedding`).

`client_name` is duplicated onto `passages`, with a composite FK to
`notes(id, client_name)`, so that the tenant predicate stays a single-table
predicate on the table actually being scanned -- a post-join tenant filter
worsens issues #12/#74.

The `vector` extension already exists by this point in the migration chain
(created in 0001); it is not re-created here.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

# revision identifiers, used by Alembic.
revision: str = '2eb0691f043c'
down_revision: Union[str, Sequence[str], None] = '57cfbdfa0d6a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "notes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("client_name", sa.Text(), nullable=False),
        sa.Column("scope", sa.Text(), nullable=False, server_default=""),
        sa.Column("path_or_uri", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("type", sa.Text(), nullable=True),
        sa.Column("provenance_kind", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=True),
        sa.Column("provenance_date", sa.Date(), nullable=True),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("pipeline_version", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("meta", JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "client_name", "scope", "path_or_uri",
            name="uq_notes_client_scope_path",
        ),
        # Exists solely so passages' composite FK can reference
        # (notes.id, notes.client_name); Postgres requires the referenced
        # columns of a composite FK to be covered by a unique constraint.
        sa.UniqueConstraint("id", "client_name", name="uq_notes_id_client"),
    )

    op.create_table(
        "passages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("note_id", sa.Integer(), nullable=False),
        sa.Column("client_name", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("heading_path", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        # Must match settings.embedding_dimension -- see notes on learnings.
        sa.Column("embedding", Vector(1024), nullable=False),
        # A plain column written by ingest -- NOT a generated column.
        sa.Column("tsv", TSVECTOR(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["note_id", "client_name"],
            ["notes.id", "notes.client_name"],
            ondelete="CASCADE",
            name="fk_passages_note",
        ),
        sa.UniqueConstraint("note_id", "ordinal", name="uq_passages_note_ordinal"),
    )

    op.create_index(
        "ix_passages_client_note", "passages", ["client_name", "note_id"]
    )
    op.execute(
        "CREATE INDEX ix_passages_tsv ON passages USING gin (tsv)"
    )

    # HNSW index for cosine distance (<=>), matching learnings_embedding_idx.
    op.execute(
        "CREATE INDEX passages_embedding_idx "
        "ON passages USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS passages_embedding_idx")
    op.execute("DROP INDEX IF EXISTS ix_passages_tsv")
    op.drop_index("ix_passages_client_note", table_name="passages")
    op.drop_table("passages")
    op.drop_table("notes")
    # Deliberately NOT dropping the vector extension -- other objects
    # (learnings) still rely on it.
