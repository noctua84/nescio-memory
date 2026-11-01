from datetime import date

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    """Request to search notes and passages."""

    q: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("q", "query"),
        description=(
            "Search query, matched by keyword/phrase and by meaning. Accepts "
            "both 'q' and 'query' aliases."
        ),
        examples=["invoice-first checkout"],
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description=(
            "Number of passages to return. Hybrid search always fills up to "
            "top_k when that many passages exist (there is no relevance "
            "floor), so trailing results can be weak matches."
        ),
    )
    repo_filter: str | None = Field(
        None,
        description="Optional filter on the note's scope.",
        examples=["project-a"],
    )

    model_config = ConfigDict(populate_by_name=True)


class SearchResult(BaseModel):
    """A passage result from a search query."""

    passage_id: int = Field(..., description="Unique identifier for the passage.")
    note_id: int = Field(..., description="Unique identifier of the note this passage belongs to.")
    ordinal: int = Field(..., description="Position of this passage within the note.")
    name: str | None = Field(
        None,
        description="Optional name of the note from frontmatter.",
    )
    description: str | None = Field(
        None,
        description="Optional description of the note from frontmatter. Contributes to FTS ranking at weight C.",
    )
    type: str | None = Field(
        None,
        description="Optional type classification from note frontmatter.",
    )
    path_or_uri: str = Field(
        ...,
        description="Unique path or URI of the note within its scope.",
    )
    heading_path: str | None = Field(
        None,
        description="Optional heading hierarchy path within the note.",
        examples=["Architecture › API Design"],
    )
    content: str = Field(
        ...,
        description="The passage text.",
    )
    provenance_kind: str | None = Field(
        None,
        description="Optional provenance source kind from the note.",
    )
    provenance_date: date | None = Field(
        None,
        description="Optional date associated with provenance from the note.",
    )
    status: str | None = Field(
        None,
        description="Optional status label from the note.",
    )
    score: float = Field(
        ...,
        description=(
            "Reciprocal-rank-fusion score; comparable only within one "
            "response; higher is more relevant."
        ),
    )
    fts_rank: int | None = Field(
        None,
        description=(
            "1-based rank in that arm's candidate list, null if the passage "
            "was not among that arm's candidates or the arm did not run. "
            "A null fts_rank means a semantic-only match."
        ),
    )
    vector_rank: int | None = Field(
        None,
        description=(
            "1-based rank in that arm's candidate list, null if the passage "
            "was not among that arm's candidates or the arm did not run."
        ),
    )


class SearchResponse(BaseModel):
    """Response containing search results."""

    results: list[SearchResult] = Field(
        ...,
        description="List of passages matching the search query, ordered by relevance.",
    )
    degraded: bool = Field(
        ...,
        description=(
            "true when the query could not be embedded and the results are "
            "full-text only"
        ),
    )
