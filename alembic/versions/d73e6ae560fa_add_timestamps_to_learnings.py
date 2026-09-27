"""add timestamps to learnings

Revision ID: d73e6ae560fa
Revises: 474ac4ba2147
Create Date: 2026-09-27 20:09:27.058226

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd73e6ae560fa'
down_revision: Union[str, Sequence[str], None] = '474ac4ba2147'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # server_default lets existing rows receive a value and NOT NULL be enforced
    # in a single step, and puts timestamp generation in the database rather than
    # in a Python expression evaluated once at import time.
    op.add_column(
        "learnings",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.add_column(
        "learnings",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("learnings", "updated_at")
    op.drop_column("learnings", "created_at")
