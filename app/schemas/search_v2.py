from datetime import date

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class SearchV2Request(BaseModel):
    """Request to search notes and passages."""

    q: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("q", "query"),
        description="Full-text search query. Accepts both 'q' and 'query' aliases.",
        examples=["invoice-first checkout"],
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Maximum number of passages to return.",
    )
    repo_filter: str | None = Field(
        None,
        description="Optional filter on the note's scope.",
        examples=["project-a"],
    )

    model_config = ConfigDict(populate_by_name=True)


class SearchV2Result(BaseModel):
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
        examples=["## Architecture > ### API Design"],
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
    rank: float = Field(
        ...,
        description="Relevance score from full-text search (higher is more relevant).",
    )


class SearchV2Response(BaseModel):
    """Response containing search results."""

    results: list[SearchV2Result] = Field(
        ...,
        description="List of passages matching the search query, ordered by relevance.",
    )
