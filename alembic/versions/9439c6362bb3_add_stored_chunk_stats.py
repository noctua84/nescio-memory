"""add stored chunk stats

Revision ID: 9439c6362bb3
Revises: 57cfbdfa0d6a
Create Date: 2026-10-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9439c6362bb3'
down_revision: Union[str, Sequence[str], None] = '57cfbdfa0d6a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Both values are computed by Postgres from the STORED column values, never
    # by application code. pgvector's float4 storage can turn a vector that
    # Python's own validation accepted into a zero-norm vector once written
    # (#42/#53), so the degenerate flag must be derived from what is actually
    # on disk, not re-derived in Python. `<=>` (cosine_distance) and
    # char_length are both IMMUTABLE (verified in Wave 0), which is required
    # for a STORED generated column.
    #
    # ADD COLUMN ... STORED rewrites the whole table once under an
    # ACCESS EXCLUSIVE lock (every row's vector is detoasted once to compute
    # the flag). That is acceptable at self-hosted sizes; on a very large
    # table, run this during a maintenance window.
    op.add_column(
        "learnings",
        sa.Column(
            "embedding_degenerate",
            sa.Boolean(),
            sa.Computed(
                "((embedding <=> embedding) = 'NaN'::float8)",
                persisted=True,
            ),
            nullable=False,
        ),
    )
    op.add_column(
        "learnings",
        sa.Column(
            "content_chars",
            sa.Integer(),
            sa.Computed(
                "char_length(content)",
                persisted=True,
            ),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("learnings", "content_chars")
    op.drop_column("learnings", "embedding_degenerate")
