"""add rate_limit_windows

Revision ID: e8d3b6f05a12
Revises: c4a1e7d29b30
Create Date: 2026-11-04 12:00:00.000000

Adds the `rate_limit_windows` table behind the per-tenant search limit. UNLOGGED:
a crash may lose the counters, which is acceptable, and every search writes here.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e8d3b6f05a12'
down_revision: str | Sequence[str] | None = 'c4a1e7d29b30'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "rate_limit_windows",
        sa.Column("client_name", sa.Text(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("client_name", "window_start"),
        prefixes=["UNLOGGED"],
    )


def downgrade() -> None:
    # Nothing is lost that a crash could not lose anyway.
    op.drop_table("rate_limit_windows")
