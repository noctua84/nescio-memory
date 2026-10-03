"""switch embedding dimension to 384 -- SUPERSEDED, the premise was wrong

Revision ID: 0002
Revises: 0001

qwen3-embedding:0.6b is 1024-dimensional, not 384: it reports
qwen3.embedding_length 1024 and /api/embeddings returns 1024 floats. The
384 figure belongs to all-MiniLM-L6-v2, the `local` backend's default, and was
applied to the Ollama default by mistake. Migration 0001 had the width right
and this migration moved away from it, leaving the default configuration unable
to embed anything at all.

Revision 57cfbdfa0d6a reverts the column to vector(1024). This file is left in
place because it has been applied in the wild; its title is corrected rather
than its behaviour. Do not take the 384 below as a statement about any Ollama
model. See issue #36.
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