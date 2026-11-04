"""Writes to ingest_events. There is no read route: operators use SQL (see the
README's Audit trail section).

Commits nothing, like NoteRepository: each event joins the transaction of the
change it records, so there is no change without a record and no record without
the change. `client_name` and `api_key_id` are stamped from self, never passed
per call, so no route can attribute an event to anyone but its caller.
"""

import hashlib
from collections.abc import Mapping, Sequence

from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from app.models.ingest_event import IngestAction, IngestEvent
from app.models.passage import Passage
from app.repositories.note import DeletedNote


def path_sha256(path_or_uri: str) -> str:
    """The stored hash of a path: plain SHA-256, unsalted, so an operator who
    knows a path can find everything that happened to it."""
    return hashlib.sha256(path_or_uri.encode("utf-8")).hexdigest()


class AuditRepository:
    def __init__(self, db: Session, client_name: str, api_key_id: int):
        self.db = db
        self.client_name = client_name
        self.api_key_id = api_key_id

    def record_ingest(
        self,
        action: IngestAction,
        scope: str,
        note_id: int,
        path_or_uri: str,
        *,
        passages_before: int | None,
        passages_after: int,
        content_hash_before: str | None,
        content_hash_after: str,
    ) -> None:
        """One event for one ingest, whichever outcome it had."""
        self.db.execute(
            insert(IngestEvent).values(
                client_name=self.client_name,
                api_key_id=self.api_key_id,
                scope=scope,
                note_id=note_id,
                action=action,
                path_or_uri=path_or_uri,
                path_sha256=path_sha256(path_or_uri),
                passages_before=passages_before,
                passages_after=passages_after,
                content_hash_before=content_hash_before,
                content_hash_after=content_hash_after,
            )
        )

    def record_deletes(
        self,
        scope: str,
        notes: Sequence[DeletedNote],
        passages_before: Mapping[int, int],
    ) -> None:
        """One event per deleted note, with its path stored hashed only.

        `passages_before` maps note id to its passage count, taken before the
        cascade removed them. A note missing from it had none.
        """
        # An executemany with no rows is not a no-op: it inserts one row of defaults.
        if not notes:
            return
        self.db.execute(
            insert(IngestEvent),
            [
                {
                    "client_name": self.client_name,
                    "api_key_id": self.api_key_id,
                    "scope": scope,
                    "note_id": note.note_id,
                    "action": "deleted",
                    "path_or_uri": None,
                    "path_sha256": path_sha256(note.path_or_uri),
                    "passages_before": passages_before.get(note.note_id, 0),
                    "passages_after": None,
                    "content_hash_before": note.content_hash,
                    "content_hash_after": None,
                }
                for note in notes
            ],
        )

    def passage_counts_in_scope(self, scope: str) -> dict[int, int]:
        """Passages per note in `scope`, for the events of a scope delete.

        NoteRepository.delete_scope reports only a total, and an event needs its
        own note's number. Like that total it is read before the DELETE, so a note
        ingested in between is deleted and recorded with 0: the count is a report,
        and the deleted list stays the authority on what went.
        """
        rows = self.db.execute(
            select(Passage.note_id, func.count())
            .where(
                Passage.client_name == self.client_name,
                Passage.scope == scope,
            )
            .group_by(Passage.note_id)
        ).all()
        return {note_id: count for note_id, count in rows}
