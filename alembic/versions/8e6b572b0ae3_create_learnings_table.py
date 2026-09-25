"""create learnings table

Revision ID: 0001
Revises:
Create Date: 2026-09-18
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Requires sufficient DB privileges to create extensions.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "learnings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("repo_name", sa.Text(), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", JSONB(), nullable=False),
        # Must match settings.embedding_dimension — see notes below.
        sa.Column("embedding", Vector(1024), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_learnings_repo_name", "learnings", ["repo_name"])

    # HNSW index for cosine distance (<=>)
    op.execute(
        "CREATE INDEX learnings_embedding_idx "
        "ON learnings USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS learnings_embedding_idx")
    op.drop_index("ix_learnings_repo_name", table_name="learnings")
    op.drop_table("learnings")
    # Deliberately NOT dropping the vector extension — other objects may rely on it.