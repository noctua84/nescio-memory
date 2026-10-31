"""drop learnings

Revision ID: 3d67961c1596
Revises: 17be352b1a4c
Create Date: 2026-10-31 00:00:00.000000

Removes the v1 `learnings` table. The v1 API was never deployed (owner
decision), so there is no production data to preserve and nothing to migrate
into `notes`/`passages`.

This is a revision of its own, rather than a step folded into the
notes/passages migrations, so that those stay independently reversible: a
downgrade past 2eb0691f043c / 17be352b1a4c must still find the `learnings`
table that 9439c6362bb3 expects.
"""
from typing import Sequence

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '3d67961c1596'
down_revision: str | Sequence[str] | None = '17be352b1a4c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The primary key, the composite btree index, the HNSW index and the
    # id sequence (owned by learnings.id) all go with the table.
    op.drop_table("learnings")


def downgrade() -> None:
    # Recreates the table EMPTY. The rows are not restored -- there is nothing
    # to restore them from, and v1 never held deployed data.
    #
    # The shape is the exact net result of 8e6b572b0ae3 (create), 474ac4ba2147
    # (client_name), d73e6ae560fa (timestamps), b6f6570b8822 + 3d80eb22a17d
    # (indexes: only the composite one survives), 57cfbdfa0d6a (vector(1024))
    # and 9439c6362bb3 (stored generated columns), in that column order.
    # Raw DDL because SQLAlchemy cannot express the STORED generated columns,
    # and the HNSW index is raw for the same reason 8e6b572b0ae3 made it raw.
    # The generation expressions are identical to 9439c6362bb3's.
    #
    # The vector extension is NOT touched: it is created by 8e6b572b0ae3 and
    # other objects (passages.embedding) rely on it.
    op.execute(
        """
        CREATE TABLE learnings (
            id serial NOT NULL,
            repo_name text NOT NULL,
            file_path text NOT NULL,
            content text NOT NULL,
            metadata jsonb NOT NULL,
            embedding vector(1024) NOT NULL,
            client_name text NOT NULL,
            created_at timestamp with time zone NOT NULL DEFAULT now(),
            updated_at timestamp with time zone NOT NULL DEFAULT now(),
            embedding_degenerate boolean
                GENERATED ALWAYS AS (((embedding <=> embedding) = 'NaN'::float8)) STORED
                NOT NULL,
            content_chars integer
                GENERATED ALWAYS AS (char_length(content)) STORED
                NOT NULL,
            CONSTRAINT learnings_pkey PRIMARY KEY (id)
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_learnings_client_repo_path "
        "ON learnings (client_name, repo_name, file_path)"
    )
    op.execute(
        "CREATE INDEX learnings_embedding_idx "
        "ON learnings USING hnsw (embedding vector_cosine_ops)"
    )
