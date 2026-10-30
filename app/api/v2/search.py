"""POST /api/v2/search -- keyword and phrase search over passages.

FTS only in this slice: no vector expression, no embedding call and no HNSW
index involvement anywhere in this module. The semantic/vector branch arrives
in slice 2; its absence here is deliberate, not an omission.
"""

from fastapi import APIRouter, Depends
from langfuse import observe
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import get_current_client
from app.models import ApiKey
from app.repositories.note import NoteRepository
from app.schemas.search_v2 import SearchV2Request, SearchV2Response, SearchV2Result

router = APIRouter()

# Input and output capture are OFF, unlike v1. Note bodies are the user's own
# memory notes and may contain personal data (the corpus has a people/ bucket),
# and Langfuse may be cloud-hosted -- @observe reads its configuration from the
# process environment, so a deployment that sets LANGFUSE_* ships whatever is
# captured off-site. Spans, timings and errors are still traced; only the
# payloads are withheld. Do not turn these back on without a privacy review.


@router.post(
    "/search",
    response_model=SearchV2Response,
    summary="Keyword and phrase search over passages",
    description=(
        "Full-text search over note passages using websearch_to_tsquery "
        "(phrases, words, exclusions). Returns individual matching passages, "
        "never whole notes, ranked by ts_rank_cd. A query that reduces to an "
        "empty tsquery (stop words only, or pure punctuation) legitimately "
        "returns an empty result list with a 200, not an error."
    ),
    operation_id="searchPassagesV2",
)
@observe(name="passage-search", capture_input=False, capture_output=False)
def search_passages(
    request: SearchV2Request,
    db: Session = Depends(get_db),
    client: ApiKey = Depends(get_current_client),
) -> SearchV2Response:
    repo = NoteRepository(db, client_name=client.client_name)
    rows = repo.search_fts(request.q, request.top_k, request.repo_filter)

    # Mapped field by field to the DTO. Never return a `Row`/model instance
    # directly -- the response shape is the contract, not whatever columns
    # search_fts happens to select.
    results = [
        SearchV2Result(
            passage_id=row.id,
            note_id=row.note_id,
            ordinal=row.ordinal,
            name=row.name,
            description=row.description,
            type=row.type,
            path_or_uri=row.path_or_uri,
            heading_path=row.heading_path,
            content=row.content,
            provenance_kind=row.provenance_kind,
            provenance_date=row.provenance_date,
            status=row.status,
            rank=row.rank,
        )
        for row in rows
    ]
    return SearchV2Response(results=results)
