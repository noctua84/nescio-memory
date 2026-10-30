import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from langfuse import observe
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.embeddings import get_embedding
from app.core.pipeline import content_hash, embedding_input, fts_lead_a, fts_lead_c, pipeline_version
from app.core.security import get_current_client
from app.core.splitter import split_note
from app.models import ApiKey
from app.repositories.note import NoteRepository, PassageRow
from app.schemas.ingest_v2 import IngestV2Request, IngestV2Response

router = APIRouter()

# Input and output capture are OFF, unlike v1. Note bodies are the user's own
# memory notes and may contain personal data (the corpus has a people/ bucket),
# and Langfuse may be cloud-hosted -- @observe reads its configuration from the
# process environment, so a deployment that sets LANGFUSE_* ships whatever is
# captured off-site. Spans, timings and errors are still traced; only the
# payloads are withheld. Do not turn these back on without a privacy review.

MAX_CONTENT_CHARS = 500_000

# Unlike v1, this route takes a JSON body (not Form(...) fields), so there is
# no Starlette per-form-field byte limit firing ahead of this check.
# app.core.body_limit.BodySizeLimitMiddleware bounds the raw request bytes
# before the body is ever buffered or parsed, including for unauthenticated
# clients; MAX_CONTENT_CHARS below is a separate, later check that bounds the
# decoded character count once the JSON body has been parsed.

_URI_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://")


def _validate_path_or_uri(path_or_uri: str) -> None:
    """Accept any URI as-is; reject an absolute path or one with '..'.

    A value matching ``scheme://`` is treated as a URI and is not subject to
    the path rule at all. Anything else is a path, stored and reflected back
    the same way v1's file_path is, so it gets the same rule: no leading '/'
    and no '..' path component.
    """
    if _URI_RE.match(path_or_uri):
        return
    if path_or_uri.startswith("/") or ".." in Path(path_or_uri).parts:
        raise HTTPException(
            status_code=400,
            detail=(
                "path_or_uri must be a relative path without '..', or an "
                "absolute URI"
            ),
        )


def _validate_content_size(content: str) -> None:
    """Reject content above the cap before anything is deleted or embedded."""
    if len(content) > MAX_CONTENT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"content is too large: {len(content)} characters exceeds the "
                f"{MAX_CONTENT_CHARS} limit."
            ),
        )


@router.post(
    "/ingest",
    response_model=IngestV2Response,
    summary="Ingest a note",
    description=(
        "Splits Markdown content into heading-aware passages, embeds them, "
        "and upserts the note and its passages. Skips re-splitting and "
        "re-embedding when the content and pipeline version are unchanged, "
        "unless force=True."
    ),
    operation_id="ingestNoteV2",
)
@observe(name="note-ingestion", capture_input=False, capture_output=False)
def ingest_note(
    request: IngestV2Request,
    db: Session = Depends(get_db),
    client: ApiKey = Depends(get_current_client),
) -> IngestV2Response:
    # Both validations run before anything is written or deleted.
    _validate_path_or_uri(request.path_or_uri)
    _validate_content_size(request.content)

    result = split_note(request.content)
    digest = content_hash(request.content)
    version = pipeline_version()

    repo = NoteRepository(db, client_name=client.client_name)

    # The FOR UPDATE lock lock_note takes is held across every get_embedding call
    # below and released only by the single commit at the end. That is
    # deliberate, and it is the same ordering v1 already has (v1 holds an open
    # transaction across its per-chunk Ollama calls -- see QWEN.md "Known Gaps",
    # item 5). Two properties come from it: the skip check happens BEFORE any
    # embedding is spent, which is the whole economy of a 574-note re-ingest; and
    # two concurrent ingests of one note serialise instead of interleaving a
    # delete with the other's insert.
    #
    # The cost is lock hold time proportional to passage count times embedding
    # latency -- for the 112 KB ADR that is tens of seconds. It is acceptable
    # because the lock is per NOTE identity and the only contender is another
    # ingest of that same note, which has nothing useful to do concurrently
    # anyway; reads never take this lock.
    #
    # If it ever does bite (concurrent ingesters of the same note timing out):
    # read the note's hash WITHOUT a lock, embed outside any transaction, then
    # open a transaction, re-take the lock, re-check the hash, and write. That
    # trades this simple invariant for a recheck and the possibility of wasted
    # embedding work. Do not reach for it before the contention is measured.
    locked = repo.lock_note(request.scope, request.path_or_uri)
    if (
        not request.force
        and locked is not None
        and locked.content_hash == digest
        and locked.pipeline_version == version
    ):
        db.commit()  # releases the lock; nothing else to write
        return IngestV2Response(
            note_id=locked.id,
            passages=repo.count_passages(locked.id),
            skipped=True,
        )

    note_id = repo.upsert_note(
        request.scope,
        request.path_or_uri,
        name=result.name,
        description=result.description,
        type_=result.type,
        provenance_kind=result.provenance_kind,
        provenance_date=result.provenance_date,
        status=request.status,
        content_hash=digest,
        pipeline_version=version,
        body=result.body,
        meta=request.meta,
    )

    rows = [
        PassageRow(
            ordinal=passage.ordinal,
            heading_path=passage.heading_path,
            content=passage.content,
            embedding=get_embedding(
                embedding_input(result.name, passage.heading_path, passage.content)
            ),
            fts_a=fts_lead_a(result.name, passage.heading_path),
            fts_c=fts_lead_c(result.description),
        )
        for passage in result.passages
    ]

    # Embedding failures (EmbeddingBackendError) propagate uncaught and are
    # mapped to 503 + Retry-After by app/core/errors.py. No commit has
    # happened yet, so a failure here leaves the previous passages intact.
    count = repo.replace_passages(note_id, rows)

    db.commit()  # single transaction boundary for the whole note
    return IngestV2Response(note_id=note_id, passages=count, skipped=False)
