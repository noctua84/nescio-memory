from datetime import datetime
from typing import Literal

from sqlalchemy import BigInteger, CheckConstraint, DateTime, Index, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

IngestAction = Literal["created", "replaced", "skipped", "deleted"]


class IngestEvent(Base):
    """One row per change to a note: who did what, and what it held before.

    Append-only: app/repositories/audit.py only inserts, and only
    scripts/prune_ingest_events.py deletes, by age. Each row is written in the
    transaction of the change it records, so neither exists without the other.
    No note content is stored here, only its hash.
    """

    __tablename__ = "ingest_events"

    __table_args__ = (
        CheckConstraint(
            "action IN ('created', 'replaced', 'skipped', 'deleted')",
            name="ck_ingest_events_action",
        ),
        # A deleted note's path is stored hashed only, and every other event
        # keeps the plain one. Enforced here so that no future writer can store
        # the path of a note whose owner asked for it to be gone.
        CheckConstraint(
            "(action = 'deleted') = (path_or_uri IS NULL)",
            name="ck_ingest_events_deleted_path_hashed",
        ),
        # The retention prune and "what happened on date D".
        Index("ix_ingest_events_at", "at"),
        # "Everything that happened to note X" and "...to path P".
        Index("ix_ingest_events_client_note_at", "client_name", "note_id", "at"),
        Index("ix_ingest_events_client_path_at", "client_name", "path_sha256", "at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # clock_timestamp(), not now(): now() is the start of the transaction, and
    # ingest's began at the auth SELECT, before it waited for the note lock and
    # embedded. Two ingests of one note could then sort in the opposite order to
    # the one they were applied in.
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.clock_timestamp(), nullable=False
    )
    client_name: Mapped[str] = mapped_column(Text, nullable=False)
    # No FK, to api_keys or to notes. An FK check would take a key-share lock on
    # the api_keys row inside ingest's long transaction, and this record has to
    # outlive both the key and the note it describes.
    api_key_id: Mapped[int] = mapped_column(Integer, nullable=False)
    scope: Mapped[str] = mapped_column(Text, nullable=False)
    note_id: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    path_or_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    path_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    # NULL means there was no such state: nothing before a `created`, nothing
    # after a `deleted`.
    passages_before: Mapped[int | None] = mapped_column(Integer, nullable=True)
    passages_after: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_hash_before: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash_after: Mapped[str | None] = mapped_column(Text, nullable=True)
