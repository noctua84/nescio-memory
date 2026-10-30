"""GET and DELETE /api/v2/notes/{note_id} -- one note by id.

Both routes share NOTE_NOT_FOUND_DETAIL and neither branches on *why* a note
did not resolve: NoteRepository.get_note/delete_note return None/False for a
missing note AND for another tenant's note, on purpose (see
app/repositories/note.py). A route that told those two cases apart would be
an existence oracle over another tenant's id space -- "404 means it never
existed, 403/other means it exists but isn't yours" leaks exactly the bit an
attacker needs to enumerate ids. So there is no such branch here; the 404 is
byte-identical (same status, body and headers) in both cases because it is
the *same* code path, not two paths that happen to agree today.
"""

from fastapi import APIRouter, Depends, HTTPException, Response, status
from langfuse import observe
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import get_current_client
from app.models import ApiKey
from app.repositories.note import NoteRepository
from app.schemas.note_v2 import NoteV2Response

router = APIRouter()

# Input and output capture are OFF, unlike v1. Note bodies are the user's own
# memory notes and may contain personal data (the corpus has a people/ bucket),
# and Langfuse may be cloud-hosted -- @observe reads its configuration from the
# process environment, so a deployment that sets LANGFUSE_* ships whatever is
# captured off-site. Spans, timings and errors are still traced; only the
# payloads are withheld. Do not turn these back on without a privacy review.

NOTE_NOT_FOUND_DETAIL = "Note not found"


@router.get(
    "/notes/{note_id}",
    response_model=NoteV2Response,
    summary="Get a note by id",
    description=(
        "Fetch one note and its passage count. Returns 404 for a note that "
        "does not exist and for another tenant's note alike -- see the "
        "module docstring."
    ),
    operation_id="getNoteV2",
    responses={404: {"description": "No such note for this client."}},
)
@observe(name="note-fetch", capture_input=False, capture_output=False)
def get_note(
    note_id: int,
    db: Session = Depends(get_db),
    client: ApiKey = Depends(get_current_client),
) -> NoteV2Response:
    repo = NoteRepository(db, client_name=client.client_name)
    row = repo.get_note(note_id)
    if row is None:
        raise HTTPException(status_code=404, detail=NOTE_NOT_FOUND_DETAIL)

    # Mapped field by field to the DTO. Never return a `Row`/model instance
    # directly -- the response shape is the contract, not whatever columns
    # get_note happens to select.
    return NoteV2Response(
        id=row.id,
        scope=row.scope,
        path_or_uri=row.path_or_uri,
        name=row.name,
        description=row.description,
        type=row.type,
        provenance_kind=row.provenance_kind,
        provenance_date=row.provenance_date,
        status=row.status,
        content_hash=row.content_hash,
        pipeline_version=row.pipeline_version,
        body=row.body,
        meta=row.meta,
        passage_count=repo.count_passages(note_id),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.delete(
    "/notes/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a note and its passages",
    description=(
        "Delete one note. Its passages go with it via cascade. Returns 404 "
        "for a note that does not exist and for another tenant's note alike "
        "-- see the module docstring."
    ),
    operation_id="deleteNoteV2",
    responses={404: {"description": "No such note for this client."}},
)
@observe(name="note-delete", capture_input=False, capture_output=False)
def delete_note(
    note_id: int,
    db: Session = Depends(get_db),
    client: ApiKey = Depends(get_current_client),
) -> Response:
    repo = NoteRepository(db, client_name=client.client_name)
    # The passages go by cascade (the composite FK's ON DELETE CASCADE, see
    # app/repositories/note.py), not by a second DELETE statement here.
    if not repo.delete_note(note_id):
        raise HTTPException(status_code=404, detail=NOTE_NOT_FOUND_DETAIL)
    # Single transaction boundary, and it runs only on the success path: a
    # 404 raised above leaves nothing to roll back.
    db.commit()

    # This DELETE is the ONLY erasure path for a note. Nothing reconciles a
    # note against a deleted source file -- there is no sweeper, no TTL, no
    # "file no longer exists on disk" check anywhere in this service. A
    # client that stops holding a file and wants its note gone must call
    # this endpoint itself.
    return Response(status_code=status.HTTP_204_NO_CONTENT)
