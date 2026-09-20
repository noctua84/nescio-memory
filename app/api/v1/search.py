from fastapi import APIRouter
from langfuse import observe

from app.core.db import get_db_connection
from app.core.embeddings import get_embedding
from app.schemas.search import SearchResponse, SearchRequest, SearchResult

router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic Memory Search",
    description="Queries pgvector using a natural language prompt.",
)
@observe(name="semantic-search")
def search_memory(request: SearchRequest):
    query_embedding = get_embedding(request.query)

    sql = """
        SELECT content, metadata, 1 - (embedding <=> %s) AS similarity
        FROM learnings
    """
    params: list = [query_embedding]

    if request.repo_filter:
        sql += " WHERE metadata->>'repo_name' = %s"
        params.append(request.repo_filter)

    sql += " ORDER BY embedding <=> %s LIMIT %s"
    params.extend([query_embedding, request.top_k])

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    results = [
        SearchResult(content=content, metadata=metadata, similarity=similarity)
        for content, metadata, similarity in rows
    ]

    return SearchResponse(results=results)