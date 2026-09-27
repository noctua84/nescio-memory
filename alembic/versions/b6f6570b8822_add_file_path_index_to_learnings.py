"""add file_path index to learnings

Revision ID: b6f6570b8822
Revises: d73e6ae560fa
Create Date: 2026-09-27 23:39:24.302653

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b6f6570b8822'
down_revision: Union[str, Sequence[str], None] = 'd73e6ae560fa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # app/models/learning.py declares index=True on file_path, but no migration
    # ever created the index. LearningRepository.delete_by_file() filters on
    # client_name, repo_name and file_path, and runs on every ingest.
    op.create_index("ix_learnings_file_path", "learnings", ["file_path"])


def downgrade() -> None:
    op.drop_index("ix_learnings_file_path", table_name="learnings")
