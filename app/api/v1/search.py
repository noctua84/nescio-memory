import logging

from fastapi import APIRouter, Depends
from langfuse import observe
from sqlalchemy.orm import Session

from app.core.context import ContextHit, FetchedChunk, build_regions, plan_ranges
from app.core.db import get_db
from app.core.embeddings import get_embedding
from app.core.security import get_current_client
from app.models import ApiKey
from app.repositories.learning import LearningRepository
from app.schemas.search import SearchResponse, SearchRequest, SearchResult

logger = logging.getLogger(__name__)

router = APIRouter()


def _chunk_index(meta: dict) -> int | None:
    """Read meta["chunk_index"] as an int, or None if it is not one.

    JSONB is free-form, so this can be missing or any JSON type. None means
    "this hit cannot be placed in a document", which costs it its context
    rather than failing the search. `bool` is excluded explicitly because it
    is an `int` subclass and a JSON `true` is not a chunk index.
    """
    index = meta.get("chunk_index")
    return index if isinstance(index, int) and not isinstance(index, bool) else None


def _expand(
    repo: LearningRepository, hits: list[ContextHit], request: SearchRequest
):
    """Fetch and build the context regions for `hits`. One query, or none."""
    ranges = plan_ranges(hits, request.context, request.context_chunks)
    fetched = [
        FetchedChunk(
            repo_name=learning.repo_name,
            file_path=learning.file_path,
            chunk_index=chunk_index,
            content=learning.content,
            chunk_size=learning.meta.get("chunk_size"),
            chunk_overlap=learning.meta.get("chunk_overlap"),
        )
        for learning, chunk_index in repo.get_chunks_for_ranges(ranges)
    ]
    return build_regions(hits, fetched, request.context, request.context_chunks)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic Memory Search",
    description="Queries pgvector using a natural language prompt.",
)
@observe(name="semantic-search")
def search_memory(request: SearchRequest, db: Session = Depends(get_db), client: ApiKey = Depends(get_current_client)):
    query_embedding = get_embedding(request.query)

    repo = LearningRepository(db, client_name=client.client_name)
    rows = repo.search(query_embedding, request.top_k, request.repo_filter)

    contexts = []
    refs: list[int | None] = [None] * len(rows)

    if request.context != "none":
        # Mapped to plain DTOs so app/core/context.py never sees a `Learning`
        # -- and so the whole expansion is testable without a database.
        hits = [
            ContextHit(
                repo_name=learning.repo_name,
                file_path=learning.file_path,
                chunk_index=_chunk_index(learning.meta),
                similarity=similarity,
            )
            for learning, similarity in rows
        ]
        try:
            contexts, refs = _expand(repo, hits, request)
        except Exception:  # noqa: BLE001 -- deliberate, see below
            # A deliberate blanket catch, not a missing `except`. The matched
            # chunks below are the primary answer to the caller's query and
            # they are already in hand; expansion is strictly additive and can
            # only ever fail to add. So anything unexpected in it -- a
            # malformed metadata shape, a database error on the extra query, a
            # bug in region building -- degrades to the same search response
            # with no context, rather than turning a successful search into a
            # non-200. Narrowing this to the exceptions anticipated today would
            # mean the ones not anticipated are exactly the ones that fail the
            # search. Logged at warning so it is visible rather than silent.
            logger.warning(
                "context expansion failed for client=%s context=%s; returning "
                "results without context",
                client.client_name,
                request.context,
                exc_info=True,
            )
            contexts, refs = [], [None] * len(rows)

    # Map ORM objects -> DTOs. Never expose `Learning` directly.
    results = [
        SearchResult(
            content=learning.content,
            metadata=learning.meta,
            similarity=similarity,
            context_ref=refs[position],
        )
        for position, (learning, similarity) in enumerate(rows)
    ]
    return SearchResponse(results=results, contexts=contexts)
