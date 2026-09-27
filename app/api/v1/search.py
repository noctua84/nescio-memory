from fastapi import APIRouter, Depends
from langfuse import observe
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.embeddings import get_embedding
from app.core.security import get_current_client
from app.models import ApiKey
from app.repositories.learning import LearningRepository
from app.schemas.search import SearchResponse, SearchRequest, SearchResult

router = APIRouter()


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

    # Map ORM objects -> DTOs. Never expose `Learning` directly.
    results = [
        SearchResult(content=l.content, metadata=l.meta, similarity=sim)
        for l, sim in rows
    ]
    return SearchResponse(results=results)