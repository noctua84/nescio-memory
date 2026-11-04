from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Body of every error response from /api/v1, other than request validation."""

    detail: str = Field(
        ...,
        description="Fixed message for the failure. Never an internal exception message.",
    )
