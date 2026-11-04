from fastapi import APIRouter, Depends, HTTPException
from langfuse import observe
from sqlalchemy.orm import Session

from app.api.v1.validation import validate_path_or_uri
from app.core.db import get_db
from app.core.embeddings import get_embeddings
from app.core.pipeline import content_hash, embedding_input, fts_lead_a, fts_lead_c, pipeline_version
from app.core.security import get_current_client
from app.core.splitter import split_note
from app.models import ApiKey
from app.repositories.note import NoteRepository, PassageRow
from app.schemas.ingest import IngestRequest, IngestResponse

router = APIRouter()

# Input and output capture are OFF, unlike the removed learnings API. Note
# bodies are the user's own
# memory notes and may contain personal data (the corpus has a people/ bucket),
# and Langfuse may be cloud-hosted -- @observe reads its configuration from the
# process environment, so a deployment that sets LANGFUSE_* ships whatever is
# captured off-site. Spans, timings and errors are still traced; only the
# payloads are withheld. Do not turn these back on without a privacy review.

MAX_CONTENT_CHARS = 500_000

# Unlike the removed learnings API, this route takes a JSON body (not
# Form(...) fields), so there is
# no Starlette per-form-field byte limit firing ahead of this check.
# app.core.body_limit.BodySizeLimitMiddleware bounds the raw request bytes
# before the body is ever buffered or parsed, including for unauthenticated
# clients; MAX_CONTENT_CHARS below is a separate, later check that bounds the
# decoded character count once the JSON body has been parsed.


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
    response_model=IngestResponse,
    summary="Ingest a note",
    description=(
        "Splits Markdown content into heading-aware passages, embeds them, "
        "and upserts the note and its passages. Skips re-splitting and "
        "re-embedding when the content and pipeline version are unchanged, "
        "unless force=True."
    ),
    operation_id="ingestNote",
)
@observe(name="note-ingestion", capture_input=False, capture_output=False)
def ingest_note(
    request: IngestRequest,
    db: Session = Depends(get_db),
    client: ApiKey = Depends(get_current_client),
) -> IngestResponse:
    # Both validations run before anything is written or deleted.
    validate_path_or_uri(request.path_or_uri)
    _validate_content_size(request.content)

    result = split_note(request.content)
    digest = content_hash(request.content)
    version = pipeline_version()

    repo = NoteRepository(db, client_name=client.client_name)

    # The FOR UPDATE lock lock_note takes is held across the batched get_embeddings call
    # below and released only by the single commit at the end. That is
    # deliberate, and it is the same ordering the removed learnings API had (it
    # held an open transaction across its per-chunk Ollama calls). Two properties
    # come from it: the skip check happens
    # BEFORE any embedding is spent, which is the whole economy of a 574-note re-ingest; and
    # two concurrent ingests of one note serialise instead of interleaving a
    # delete with the other's insert.
    #
    # The cost is lock hold time of about ceil(passages / embed_batch_size)
    # embedding round-trips -- one per sub-batch, not one per passage -- plus,
    # when the backend is flaky, the retries: at most EMBED_RETRY_BUDGET of them
    # for the whole note (shared across sub-batches), each costing up to one
    # 30 s call and EMBED_RETRY_MAX_DELAY_S of backoff. It is acceptable
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
        return IngestResponse(
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

    # One call for the whole note; vectors come back in input order, so they are
    # zipped onto the passages by position. strict=True turns a count mismatch
    # into an error instead of silently dropping passages.
    vectors = get_embeddings(
        [
            embedding_input(result.name, passage.heading_path, passage.content)
            for passage in result.passages
        ]
    )
    rows = [
        PassageRow(
            ordinal=passage.ordinal,
            heading_path=passage.heading_path,
            content=passage.content,
            embedding=vector,
            fts_a=fts_lead_a(result.name, passage.heading_path),
            fts_c=fts_lead_c(result.description),
        )
        for passage, vector in zip(result.passages, vectors, strict=True)
    ]

    # Embedding failures (EmbeddingBackendError) propagate uncaught and are
    # mapped to 503 + Retry-After by app/core/errors.py. No commit has
    # happened yet, so a failure here leaves the previous passages intact.
    # Transient ones were already retried inside get_embeddings, within the
    # budget described above the lock.
    count = repo.replace_passages(note_id, request.scope, rows)

    db.commit()  # single transaction boundary for the whole note
    return IngestResponse(note_id=note_id, passages=count, skipped=False)
