"""composite learnings index

Revision ID: 3d80eb22a17d
Revises: b6f6570b8822
Create Date: 2026-09-30 16:10:06.067851

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3d80eb22a17d'
down_revision: Union[str, Sequence[str], None] = 'b6f6570b8822'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Created first, so no window exists where none of these columns is indexed.
    op.create_index(
        "ix_learnings_client_repo_path",
        "learnings",
        ["client_name", "repo_name", "file_path"],
    )
    # Now redundant: the composite's prefixes cover client_name alone and
    # client_name with repo_name, and file_path is never filtered on its own.
    op.drop_index("ix_learnings_client_name", table_name="learnings")
    op.drop_index("ix_learnings_repo_name", table_name="learnings")
    op.drop_index("ix_learnings_file_path", table_name="learnings")


def downgrade() -> None:
    op.create_index("ix_learnings_client_name", "learnings", ["client_name"])
    op.create_index("ix_learnings_repo_name", "learnings", ["repo_name"])
    op.create_index("ix_learnings_file_path", "learnings", ["file_path"])
    op.drop_index("ix_learnings_client_repo_path", table_name="learnings")
