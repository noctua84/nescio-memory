from datetime import date, datetime

from pydantic import BaseModel, Field


class NoteV2Response(BaseModel):
    """Response containing a note and its passage count."""

    id: int = Field(..., description="Unique identifier for the note.")
    scope: str = Field(
        ...,
        description="Scope to organize notes within a tenant.",
        examples=["project-a", ""],
    )
    path_or_uri: str = Field(
        ...,
        description="Unique path or URI of the note within its scope.",
        examples=["docs/architecture/adr-001.md"],
    )
    name: str | None = Field(
        None,
        description="Optional human-readable name extracted from frontmatter.",
        examples=["ADR-001: Use PostgreSQL"],
    )
    description: str | None = Field(
        None,
        description="Optional summary from frontmatter. Used in search at FTS weight C.",
    )
    type: str | None = Field(
        None,
        description="Optional type classification from frontmatter (e.g., 'adr', 'design-doc').",
        examples=["adr"],
    )
    provenance_kind: str | None = Field(
        None,
        description="Optional provenance source kind (e.g., 'user override', 'empirical', 'agent inference').",
        examples=["user override"],
    )
    provenance_date: date | None = Field(
        None,
        description="Optional date associated with provenance.",
    )
    status: str | None = Field(
        None,
        description="Optional status label (e.g., 'approved', 'superseded').",
        examples=["approved"],
    )
    content_hash: str = Field(
        ...,
        description="Hash of the note content, used to detect changes.",
    )
    pipeline_version: str = Field(
        ...,
        description="Version identifier combining splitter, FTS composition, embedding-input composition, and embedding backend+dimension.",
    )
    body: str = Field(
        ...,
        description="Full markdown content of the note.",
    )
    meta: dict = Field(
        ...,
        description="Optional metadata dictionary.",
    )
    passage_count: int = Field(
        ...,
        description="Number of passages this note has been split into.",
    )
    created_at: datetime = Field(..., description="Timestamp when the note was created.")
    updated_at: datetime = Field(..., description="Timestamp when the note was last updated.")
