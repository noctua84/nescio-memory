from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        description="The natural language search query.",
        examples=["How do we handle API authentication across repos?"],
        validation_alias="query",
    )
    top_k: int = Field(5, ge=1, le=50, description="Max chunks to return.")
    repo_filter: str | None = Field(
        None,
        description="Optional filter by repository name.",
        examples=["repo_a"],
        validation_alias="repo_filter",
    )


class SearchResult(BaseModel):
    content: str = Field(..., description="The retrieved markdown chunk.")
    metadata: dict = Field(..., description="File path, repo name, chunk index.")
    similarity: float = Field(..., description="Cosine similarity score (0-1).")


class SearchResponse(BaseModel):
    results: list[SearchResult]