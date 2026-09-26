"""switch embedding dimension to 384 (qwen3-embedding-0.6b)

Revision ID: 0002
Revises: 0001
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing 1024-dim embeddings CANNOT be cast to 384-dim.
    # They must be dropped and re-embedded with the new model.
    op.execute("TRUNCATE TABLE learnings")

    # Drop the HNSW index (it's bound to the old dimension)
    op.execute("DROP INDEX IF EXISTS learnings_embedding_idx")

    # Change the column dimension
    op.execute("ALTER TABLE learnings ALTER COLUMN embedding TYPE vector(384)")

    # Rebuild the HNSW index for the new dimension
    op.execute(
        "CREATE INDEX learnings_embedding_idx "
        "ON learnings USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    # Lossy: restores the schema only, not the truncated data.
    op.execute("TRUNCATE TABLE learnings")
    op.execute("DROP INDEX IF EXISTS learnings_embedding_idx")
    op.execute("ALTER TABLE learnings ALTER COLUMN embedding TYPE vector(1024)")
    op.execute(
        "CREATE INDEX learnings_embedding_idx "
        "ON learnings USING hnsw (embedding vector_cosine_ops)"
    )