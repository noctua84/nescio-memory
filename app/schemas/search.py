from typing import Literal

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
    context: Literal["none", "neighbors", "document"] = Field(
        "none",
        description=(
            "How much surrounding text to return alongside each matched "
            "chunk. \"none\" (the default) returns matched chunks only and "
            "runs no extra query. \"neighbors\" expands each hit by "
            "`context_chunks` chunks either side. \"document\" expands to the "
            "whole note the hit came from. Expanded text is returned in the "
            "response's `contexts` list, not inline, so two hits in the same "
            "note share one region."
        ),
    )
    context_chunks: int = Field(
        1,
        ge=1,
        le=5,
        description=(
            "Chunks to include either side of a hit. Only meaningful when "
            "`context` is \"neighbors\" -- ignored for \"none\" and "
            "\"document\"."
        ),
    )


# Field names below are snake_case with no serialization_alias, matching
# SearchResult's existing style rather than IngestResponse's aliased one.
class ContextRegion(BaseModel):
    repo_name: str = Field(..., description="Repository the region's note belongs to.")
    file_path: str = Field(..., description="Relative path of the region's note.")
    # Nullable on purpose. A hole in the chunk index sequence leaves the
    # dropped chunk's characters in no neighbouring chunk, so there is no safe
    # text to join at all; null says so, where "" would look like a genuinely
    # empty passage.
    content: str | None = Field(
        None,
        description=(
            "The expanded text, with the chunks' shared overlap stripped at "
            "each join. Null when the stored chunks admit no safe join (see "
            "`note`)."
        ),
    )
    chunk_index_from: int = Field(
        ..., description="Lowest chunk index included in this region."
    )
    chunk_index_to: int = Field(
        ..., description="Highest chunk index included in this region."
    )
    covers: str = Field(
        ...,
        description=(
            "The `context` mode this region was requested under: "
            "\"neighbors\" or \"document\"."
        ),
    )
    exact: bool = Field(
        ...,
        description=(
            "Whether this region provably reproduces the corresponding span "
            "of the original note. False means the text is best-effort and "
            "`note` says why."
        ),
    )
    note: str | None = Field(
        None,
        description=(
            "Why the region is not exactly what was asked for -- truncated at "
            "a dropped chunk, or joined under an unverifiable window. Null "
            "when there is nothing to report."
        ),
    )


class SearchResult(BaseModel):
    content: str = Field(..., description="The retrieved markdown chunk.")
    metadata: dict = Field(..., description="File path, repo name, chunk index.")
    similarity: float = Field(..., description="Cosine similarity score (0-1).")
    context_ref: int | None = Field(
        None,
        description=(
            "Index into the response's `contexts` list of the region "
            "containing this chunk. Null when no region was built for it -- "
            "`context` was \"none\", the note's chunks could not be resolved, "
            "or the region did not fit the response's character budget."
        ),
    )


class SearchResponse(BaseModel):
    results: list[SearchResult]
    contexts: list[ContextRegion] = Field(
        default_factory=list,
        description=(
            "Expanded regions referenced by the results' `context_ref`. "
            "Deduplicated and merged: two hits in the same note share one "
            "region rather than repeating its text. Empty when `context` is "
            "\"none\" or when expansion produced nothing."
        ),
    )
