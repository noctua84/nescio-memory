"""add last_used_at and ingest_events

Revision ID: c4a1e7d29b30
Revises: 3d67961c1596
Create Date: 2026-11-04 00:00:00.000000

Adds `api_keys.last_used_at` and the append-only `ingest_events` table: a
durable record of who changed which note, readable without Langfuse.
"""
from typing import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c4a1e7d29b30'
down_revision: str | Sequence[str] | None = '3d67961c1596'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "api_keys",
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
    )

    # api_key_id and note_id are plain integers with no FK, on purpose: an FK to
    # api_keys would take a key-share lock on the key row inside ingest's long
    # transaction, and an FK to notes could not survive the delete it records.
    op.create_table(
        "ingest_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "at",
            sa.DateTime(timezone=True),
            server_default=sa.text("clock_timestamp()"),
            nullable=False,
        ),
        sa.Column("client_name", sa.Text(), nullable=False),
        sa.Column("api_key_id", sa.Integer(), nullable=False),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column("note_id", sa.Integer(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("path_or_uri", sa.Text(), nullable=True),
        sa.Column("path_sha256", sa.Text(), nullable=False),
        sa.Column("passages_before", sa.Integer(), nullable=True),
        sa.Column("passages_after", sa.Integer(), nullable=True),
        sa.Column("content_hash_before", sa.Text(), nullable=True),
        sa.Column("content_hash_after", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "action IN ('created', 'replaced', 'skipped', 'deleted')",
            name="ck_ingest_events_action",
        ),
        sa.CheckConstraint(
            "(action = 'deleted') = (path_or_uri IS NULL)",
            name="ck_ingest_events_deleted_path_hashed",
        ),
    )
    op.create_index("ix_ingest_events_at", "ingest_events", ["at"])
    op.create_index(
        "ix_ingest_events_client_note_at",
        "ingest_events",
        ["client_name", "note_id", "at"],
    )
    op.create_index(
        "ix_ingest_events_client_path_at",
        "ingest_events",
        ["client_name", "path_sha256", "at"],
    )


def downgrade() -> None:
    # Dropping the table drops its indexes and constraints with it. The recorded
    # history goes too: there is nowhere else to keep it.
    op.drop_table("ingest_events")
    op.drop_column("api_keys", "last_used_at")
