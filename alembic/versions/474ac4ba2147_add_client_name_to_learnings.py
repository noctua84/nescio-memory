"""add client_name to learnings

Revision ID: 474ac4ba2147
Revises: 8f51a956ada9
Create Date: 2026-09-26 15:13:59.029157

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '474ac4ba2147'
down_revision: Union[str, Sequence[str], None] = '8f51a956ada9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Existing rows need a value before we enforce NOT NULL.
LEGACY_CLIENT_NAME = "default"   # change if you prefer, or re-ingest


def upgrade() -> None:
    op.add_column("learnings", sa.Column("client_name", sa.Text(), nullable=True))
    op.execute(
        f"UPDATE learnings SET client_name = '{LEGACY_CLIENT_NAME}' "
        "WHERE client_name IS NULL"
    )
    op.alter_column("learnings", "client_name", nullable=False)
    op.create_index("ix_learnings_client_name", "learnings", ["client_name"])


def downgrade() -> None:
    op.drop_index("ix_learnings_client_name", table_name="learnings")
    op.drop_column("learnings", "client_name")
