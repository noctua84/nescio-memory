"""add stored chunk stats

Revision ID: 9439c6362bb3
Revises: 57cfbdfa0d6a
Create Date: 2026-10-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


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
    # A single ALTER TABLE with two ADD COLUMNs, issued as one statement
    # rather than as two op.add_column() calls. Each ADD COLUMN ... STORED
    # rewrites the whole table under an ACCESS EXCLUSIVE lock -- every row is
    # read and rewritten to compute the new column -- and, because this table
    # also carries the HNSW index `learnings_embedding_idx`, that rewrite
    # rebuilds every index on the table too (confirmed by relfilenode
    # tracking: both the heap and the HNSW index get a new filenode). The
    # index rebuild is the dominant cost here, not the column computation --
    # an HNSW build is far more expensive per row than evaluating `<=>` or
    # char_length. Two separate op.add_column() calls would rewrite the heap
    # and rebuild the index twice instead of once, doubling the lock time for
    # no benefit, so both columns are added in the same ALTER TABLE
    # statement. That is acceptable at self-hosted sizes; on a very large
    # table, run this during a maintenance window.
    #
    # WIDTH CHANGE WARNING: a future migration that changes embedding's
    # vector width (as 57cfbdfa0d6a and 58f3ecef1edf did) cannot run
    # `ALTER COLUMN embedding TYPE vector(N)` while embedding_degenerate
    # exists -- Postgres refuses to alter the type of a column a generated
    # column reads. Such a migration must DROP embedding_degenerate first,
    # alter embedding's type, then re-add embedding_degenerate with this same
    # generation expression.
    op.execute(
        """
        ALTER TABLE learnings
            ADD COLUMN embedding_degenerate boolean
                GENERATED ALWAYS AS (((embedding <=> embedding) = 'NaN'::float8)) STORED
                NOT NULL,
            ADD COLUMN content_chars integer
                GENERATED ALWAYS AS (char_length(content)) STORED
                NOT NULL
        """
    )


def downgrade() -> None:
    # Also a single ALTER TABLE: unlike ADD COLUMN ... STORED, DROP COLUMN
    # does not rewrite the table (it just marks the column dropped in the
    # catalog and leaves the on-disk data to be reclaimed by VACUUM), so
    # there is no rewrite cost to double here. Combined into one statement
    # anyway, for symmetry with upgrade() and to hold the lock once rather
    # than twice.
    op.execute(
        """
        ALTER TABLE learnings
            DROP COLUMN content_chars,
            DROP COLUMN embedding_degenerate
        """
    )
