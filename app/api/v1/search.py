"""POST /api/v1/search -- hybrid keyword, phrase and semantic search over passages.

Two arms (full-text and vector) are fused by reciprocal rank in the repository.
With VECTOR_WEIGHT=0 the vector arm is off and the query is never embedded.
"""

import logging

from fastapi import APIRouter, Depends
from langfuse import observe, propagate_attributes
from sqlalchemy.orm import Session

from app.config import settings
from app.core.db import get_db
from app.core.embeddings import get_query_embedding
from app.core.errors import EmbeddingBackendError
from app.core.security import get_current_client
from app.core.tracing import annotate_span
from app.models import ApiKey
from app.repositories.passage_search import PassageSearchRepository
from app.schemas.search import SearchRequest, SearchResponse, SearchResult

logger = logging.getLogger(__name__)

router = APIRouter()

# Input and output capture are OFF, unlike the removed learnings API. Note
# bodies are the user's own
# memory notes and may contain personal data (the corpus has a people/ bucket),
# and Langfuse may be cloud-hosted -- @observe reads its configuration from the
# process environment, so a deployment that sets LANGFUSE_* ships whatever is
# captured off-site. Spans, timings and errors are still traced; only the
# payloads are withheld. Do not turn these back on without a privacy review.


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Hybrid keyword and semantic search over passages",
    description=(
        "Hybrid search over note passages: a full-text arm "
        "(websearch_to_tsquery: phrases, words, exclusions) and a vector arm "
        "(semantic similarity), fused by reciprocal rank. Returns individual "
        "matching passages, never whole notes, ordered by the fused score. "
        "Always fills up to top_k when that many passages exist -- there is no "
        "relevance floor -- and fts_rank null marks a semantic-only match. A "
        "query with no usable keywords (stop words only) is answered by the "
        "vector arm. If the query cannot be embedded because the embedding "
        "backend is briefly unavailable, the response is full-text only and "
        "degraded is true."
    ),
    operation_id="searchPassages",
)
@observe(name="passage-search", capture_input=False, capture_output=False)
def search_passages(
    request: SearchRequest,
    db: Session = Depends(get_db),
    client: ApiKey = Depends(get_current_client),
) -> SearchResponse:
    # Trace attribution; the query text is never attached. A None value would be
    # dropped with a warning on every unfiltered search, hence the condition.
    metadata = {} if request.repo_filter is None else {"scope": request.repo_filter}
    with propagate_attributes(user_id=client.client_name, metadata=metadata):
        response = _search(request, db, client)
        annotate_span(results=len(response.results), degraded=response.degraded)
    return response


def _search(request: SearchRequest, db: Session, client: ApiKey) -> SearchResponse:
    # Embed BEFORE the repository's first statement so no search work (and no
    # statement_timeout window) overlaps the HTTP call. get_current_client has
    # already run a SELECT on this session, which autobegins a transaction and
    # checks a pool connection out, and the embed can wait up to
    # QUERY_EMBED_TIMEOUT_S on Ollama (a cold model load is the slow case). So
    # the auth read is ended first, or every in-flight embed would pin one
    # idle-in-transaction connection and a slow Ollama would exhaust the pool.
    # commit(), not rollback()/close(): it ends the read-only transaction and
    # returns the connection to the pool, and under the test fixture it releases
    # only the SAVEPOINT, whereas rollback would discard the seeded rows.
    # client_name is read first because commit() expires the ORM instance. The
    # repository's first statement re-begins a transaction, which its set_config
    # read-back guard relies on.
    # VECTOR_WEIGHT=0 is the FTS-only baseline mode: the embedder is never called.
    client_name = client.client_name
    db.commit()
    query_vector: list[float] | None = None
    degraded = False
    if settings.vector_weight > 0:
        try:
            query_vector = get_query_embedding(request.q)
        except EmbeddingBackendError as exc:
            # Exact class only. The base class means the backend is transiently
            # unreachable, where full-text results are a useful answer. The
            # subclasses mean it answered wrongly (bad response) or we are
            # misconfigured (wrong model/dimension); serving FTS-only there
            # would hide a fault that needs attention, so they propagate to the
            # 503/500 handlers in app/core/errors.py. Ingest does NOT degrade:
            # it still returns 503, since a note stored without vectors would
            # be silently unsearchable by meaning.
            if type(exc) is not EmbeddingBackendError:
                raise
            # The message is safe to log: on this path (exact base class) it is
            # built only from the HTTP status, httpx's transport error text, or
            # the local model's runtime error -- never the query. The query
            # text is in the request body, which none of them echo. Without the
            # message the log cannot tell a timeout from a reset connection.
            logger.warning(
                "query embedding unavailable (%s: %s); serving full-text "
                "results only",
                type(exc).__name__,
                exc,
            )
            degraded = True

    repo = PassageSearchRepository(db, client_name=client_name)
    rows = repo.search(request.q, query_vector, request.top_k, request.repo_filter)

    # Mapped field by field to the DTO. Never return a `Row`/model instance
    # directly -- the response shape is the contract, not whatever columns
    # the search happens to select.
    results = [
        SearchResult(
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
            score=row.score,
            fts_rank=row.fts_rank,
            vector_rank=row.vector_rank,
        )
        for row in rows
    ]
    return SearchResponse(results=results, degraded=degraded)
