"""restore embedding dimension to 1024 (qwen3-embedding-0.6b)

Revision ID: 57cfbdfa0d6a
Revises: 3d80eb22a17d
Create Date: 2026-10-03 07:35:45.608872

Reverts migration 0002, which narrowed this column to vector(384) on the
mistaken premise that qwen3-embedding:0.6b is 384-dimensional. It is 1024: the
model reports qwen3.embedding_length 1024 and /api/embeddings returns a
1024-element vector. Migration 0001 had it right. See issue #36.

A RE-INGEST IS REQUIRED. 384-dim vectors cannot be cast to 1024, so any rows a
deployment ingested under the old configuration are deleted here and must be
re-embedded from source. In practice such a deployment could not have ingested
anything at all -- the dimension check in app.core.embeddings rejected every
embedding the default backend produced -- but a deployment running the `local`
backend at 384 genuinely does lose its rows.

Operators who want to keep running the `local` backend
(all-MiniLM-L6-v2, 384-dim) should stay on revision 3d80eb22a17d instead of
upgrading to this one, and leave EMBEDDING_DIMENSION=384.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '57cfbdfa0d6a'
down_revision: Union[str, Sequence[str], None] = '3d80eb22a17d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Existing 384-dim embeddings CANNOT be cast to 1024-dim.
    # They must be dropped and re-embedded. See the note above.
    op.execute("TRUNCATE TABLE learnings")

    # The HNSW index has to go first: its vector_cosine_ops operator class is
    # bound to the column's declared dimension, so ALTER TYPE cannot rewrite
    # the column underneath it.
    op.execute("DROP INDEX IF EXISTS learnings_embedding_idx")

    op.execute("ALTER TABLE learnings ALTER COLUMN embedding TYPE vector(1024)")

    # Rebuild at the new dimension. Cheap on an empty table, which is why the
    # truncate above comes first.
    op.execute(
        "CREATE INDEX learnings_embedding_idx "
        "ON learnings USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    # Lossy in the same way and for the same reason: restores the vector(384)
    # schema that migration 0002 left behind, not the truncated rows.
    op.execute("TRUNCATE TABLE learnings")
    op.execute("DROP INDEX IF EXISTS learnings_embedding_idx")
    op.execute("ALTER TABLE learnings ALTER COLUMN embedding TYPE vector(384)")
    op.execute(
        "CREATE INDEX learnings_embedding_idx "
        "ON learnings USING hnsw (embedding vector_cosine_ops)"
    )
