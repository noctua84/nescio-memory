"""GET and DELETE /api/v1/notes/{note_id} -- one note by id -- plus delete by
path and delete by scope.

Both routes share NOTE_NOT_FOUND_DETAIL and neither branches on *why* a note
did not resolve: NoteRepository.get_note/delete_note return None for a
missing note AND for another tenant's note, on purpose (see
app/repositories/note.py). A route that told those two cases apart would be
an existence oracle over another tenant's id space -- "404 means it never
existed, 403/other means it exists but isn't yours" leaks exactly the bit an
attacker needs to enumerate ids. So there is no such branch here; the 404 is
byte-identical (same status, body and headers) in both cases because it is
the *same* code path, not two paths that happen to agree today.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from langfuse import observe, propagate_attributes
from sqlalchemy.orm import Session

from app.api.v1.validation import validate_path_or_uri
from app.core.db import get_db
from app.core.security import get_current_client
from app.core.tracing import annotate_span
from app.models import ApiKey
from app.repositories.audit import AuditRepository
from app.repositories.note import DeletedNote as RepoDeletedNote
from app.repositories.note import NoteRepository
from app.schemas.errors import ErrorDetail
from app.schemas.note import DeleteNotesResponse, DeletedNote, NoteResponse

router = APIRouter()

# Input and output capture are OFF, unlike the removed learnings API. Note
# bodies are the user's own
# memory notes and may contain personal data (the corpus has a people/ bucket),
# and Langfuse may be cloud-hosted -- @observe reads its configuration from the
# process environment, so a deployment that sets LANGFUSE_* ships whatever is
# captured off-site. Spans, timings and errors are still traced; only the
# payloads are withheld. Do not turn these back on without a privacy review.
# The delete routes attach who and how much to their trace, never path_or_uri.

NOTE_NOT_FOUND_DETAIL = "Note not found"

# The two 400s of DELETE /scopes. Constants because the tests pin the exact
# text, and so a client can match on them.
SCOPE_REQUIRED_DETAIL = (
    "scope is required. To delete the root (empty) scope, send "
    "root_scope=true with no scope."
)
ROOT_SCOPE_CONFLICT_DETAIL = (
    "root_scope=true deletes the root (empty) scope and cannot be combined "
    "with a non-empty scope."
)


@router.get(
    "/notes/{note_id}",
    response_model=NoteResponse,
    summary="Get a note by id",
    description=(
        "Fetch one note and its passage count. Returns 404 for a note that "
        "does not exist and for another tenant's note alike -- see the "
        "module docstring."
    ),
    operation_id="getNote",
    responses={404: {"description": "No such note for this client."}},
)
@observe(name="note-fetch", capture_input=False, capture_output=False)
def get_note(
    note_id: int,
    db: Session = Depends(get_db),
    client: ApiKey = Depends(get_current_client),
) -> NoteResponse:
    repo = NoteRepository(db, client_name=client.client_name)
    row = repo.get_note(note_id)
    if row is None:
        raise HTTPException(status_code=404, detail=NOTE_NOT_FOUND_DETAIL)

    # Mapped field by field to the DTO. Never return a `Row`/model instance
    # directly -- the response shape is the contract, not whatever columns
    # get_note happens to select.
    return NoteResponse(
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
    operation_id="deleteNote",
    responses={404: {"description": "No such note for this client."}},
)
@observe(name="note-delete", capture_input=False, capture_output=False)
def delete_note(
    note_id: int,
    db: Session = Depends(get_db),
    client: ApiKey = Depends(get_current_client),
) -> Response:
    with propagate_attributes(
        user_id=client.client_name, metadata={"note_id": note_id}
    ):
        repo = NoteRepository(db, client_name=client.client_name)
        # Counted first: the passages go by cascade (the composite FK's ON DELETE
        # CASCADE, see app/repositories/note.py), not by a second DELETE statement
        # here, so afterwards there is nothing left to count.
        passages_before = repo.count_passages(note_id)
        row = repo.delete_note(note_id)
        if row is None:
            raise HTTPException(status_code=404, detail=NOTE_NOT_FOUND_DETAIL)
        _audit(db, client).record_deletes(
            row.scope,
            [RepoDeletedNote(row.id, row.path_or_uri, row.content_hash)],
            {row.id: passages_before},
        )
        # Single transaction boundary, and it runs only on the success path: a
        # 404 raised above leaves nothing to roll back.
        db.commit()
        annotate_span(scope=row.scope, passages_deleted=passages_before)

    # Deletes (this one, by path, by scope) are the ONLY erasure paths for a
    # note. Nothing reconciles a note against a deleted source file -- there is
    # no sweeper, no TTL, no "file no longer exists on disk" check anywhere in
    # this service. A client that stops holding a file and wants its note gone
    # must call one of them itself.
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/notes",
    response_model=DeleteNotesResponse,
    summary="Delete a note by scope and path",
    description=(
        "Delete the note stored at (scope, path_or_uri) and, by cascade, its "
        "passages -- for a client that does not hold the note id. Both query "
        "parameters are required; scope may be the empty string, which is the "
        "root scope. Returns 404 when there is no such note for this client, "
        "and for another tenant's note alike."
    ),
    operation_id="deleteNoteByPath",
    responses={
        400: {
            "model": ErrorDetail,
            "description": "path_or_uri is an absolute path or contains '..'.",
        },
        404: {"description": "No such note for this client."},
    },
)
@observe(name="note-delete-by-path", capture_input=False, capture_output=False)
def delete_note_by_path(
    # Query parameters, not path segments: scope may be "" and a path_or_uri
    # contains slashes. `Query(...)` makes the parameter required while still
    # accepting an empty value, which is what lets the root scope be named.
    scope: str = Query(..., max_length=512, description="Scope of the note; may be empty."),
    path_or_uri: str = Query(
        ..., min_length=1, max_length=2048, description="Path or URI of the note within its scope."
    ),
    db: Session = Depends(get_db),
    client: ApiKey = Depends(get_current_client),
) -> DeleteNotesResponse:
    with propagate_attributes(user_id=client.client_name, metadata={"scope": scope}):
        validate_path_or_uri(path_or_uri)

        repo = NoteRepository(db, client_name=client.client_name)
        audit = _audit(db, client)
        deleted, passages_deleted = repo.delete_by_path(scope, path_or_uri)
        # Same 404 as delete by id, and for the same reason it is not an oracle:
        # the lookup is tenant-scoped, so "no match" and "another tenant's note"
        # are one code path.
        if not deleted:
            raise HTTPException(status_code=404, detail=NOTE_NOT_FOUND_DETAIL)
        # (scope, path_or_uri) is unique per client, so the total is this note's.
        audit.record_deletes(scope, deleted, {deleted[0].note_id: passages_deleted})
        # Single transaction boundary, success path only, as in delete_note.
        db.commit()
        return _traced_delete_response(deleted, passages_deleted)


@router.delete(
    "/scopes",
    response_model=DeleteNotesResponse,
    summary="Delete every note in a scope",
    description=(
        "Retire a whole scope: delete every note in it and, by cascade, their "
        "passages. An empty or omitted scope is refused with 400 unless "
        "root_scope=true is sent, in which case scope must be empty or "
        "omitted and the root (empty) scope is the one deleted. A scope with "
        "no notes is not an error: the response is 200 with zero counts."
    ),
    operation_id="deleteScope",
    responses={
        400: {
            "model": ErrorDetail,
            "description": (
                "scope is empty or omitted without root_scope=true, or "
                "root_scope=true was combined with a non-empty scope."
            )
        },
    },
)
@observe(name="scope-delete", capture_input=False, capture_output=False)
def delete_scope(
    scope: str | None = Query(
        None, max_length=512, description="Scope to delete. Omit or leave empty only with root_scope=true."
    ),
    root_scope: bool = Query(
        False,
        description=(
            "Must be true to delete the root (empty) scope; then scope must "
            "be empty or omitted."
        ),
    ),
    db: Session = Depends(get_db),
    client: ApiKey = Depends(get_current_client),
) -> DeleteNotesResponse:
    # The root scope ("") is real data -- an ingest without a scope lands
    # there -- and a client that builds this URL from a variable which turned
    # out empty would otherwise erase it. Nobody attends this service, so the
    # only guard is this one: the destructive default has to be spelled out.
    # Both checks run before any statement, so a refusal touches nothing.
    if root_scope:
        if scope:
            raise HTTPException(status_code=400, detail=ROOT_SCOPE_CONFLICT_DETAIL)
    elif not scope:
        raise HTTPException(status_code=400, detail=SCOPE_REQUIRED_DETAIL)

    target_scope = scope or ""
    with propagate_attributes(
        user_id=client.client_name, metadata={"scope": target_scope}
    ):
        repo = NoteRepository(db, client_name=client.client_name)
        audit = _audit(db, client)
        passages_per_note = audit.passage_counts_in_scope(target_scope)
        deleted, passages_deleted = repo.delete_scope(target_scope)
        # No 404 for an empty scope: "nothing to delete" is the state the caller
        # asked for, and 200 with zero counts keeps a retry after a partial
        # failure idempotent. It is tenant-scoped, so it cannot reveal another
        # tenant's scope either. No event either: nothing happened.
        audit.record_deletes(target_scope, deleted, passages_per_note)
        db.commit()
        return _traced_delete_response(deleted, passages_deleted)


def _audit(db: Session, client: ApiKey) -> AuditRepository:
    # Read while the transaction is open: the commit expires the ORM instance.
    return AuditRepository(db, client_name=client.client_name, api_key_id=client.id)


def _traced_delete_response(
    deleted: list[RepoDeletedNote], passages_deleted: int
) -> DeleteNotesResponse:
    metadata = {"notes_deleted": len(deleted), "passages_deleted": passages_deleted}
    if len(deleted) == 1:
        metadata["note_id"] = deleted[0].note_id
    annotate_span(**metadata)
    return _delete_response(deleted, passages_deleted)


def _delete_response(deleted: list[RepoDeletedNote], passages_deleted: int) -> DeleteNotesResponse:
    # Mapped field by field to the DTO, like get_note.
    return DeleteNotesResponse(
        notes_deleted=len(deleted),
        passages_deleted=passages_deleted,
        notes=[
            DeletedNote(
                note_id=note.note_id,
                path_or_uri=note.path_or_uri,
                content_hash=note.content_hash,
            )
            for note in deleted
        ],
    )
