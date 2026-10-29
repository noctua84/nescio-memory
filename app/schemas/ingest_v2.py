from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class IngestV2Request(BaseModel):
    """Request to ingest a note with passages."""

    scope: str = Field(
        "",
        validation_alias=AliasChoices("scope", "repo_name"),
        max_length=512,
        description="Scope to organize notes within a tenant. Accepts both 'scope' and 'repo_name' aliases for backwards compatibility.",
        examples=["project-a", ""],
    )
    path_or_uri: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Unique path or URI of the note within its scope.",
        examples=["docs/architecture/adr-001.md"],
    )
    content: str = Field(
        ...,
        description="Full markdown content of the note.",
    )
    status: str | None = Field(
        None,
        max_length=256,
        description="Optional status label for the note (e.g., 'draft', 'approved', 'superseded').",
        examples=["approved"],
    )
    meta: dict = Field(
        default_factory=dict,
        description="Optional metadata dictionary for the note.",
    )
    force: bool = Field(
        False,
        description=(
            "Re-split, re-embed and rewrite this note's passages even when "
            "the content hash and pipeline version both match. The escape "
            "hatch for a change the pipeline version cannot see -- most "
            "plausibly a re-pulled embedding model behind an unchanged "
            "backend name and dimension."
        ),
    )

    model_config = ConfigDict(populate_by_name=True)


class IngestV2Response(BaseModel):
    """Response after ingesting a note."""

    note_id: int = Field(..., description="ID of the ingested note.")
    passages: int = Field(..., description="Number of passages now stored for this note.")
    skipped: bool = Field(
        ...,
        description="True when the skip fired (content unchanged and force=False); always False under force.",
    )
