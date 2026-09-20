from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    status: str = Field(..., description="Operation status.", examples=["success"], serialization_alias="status")
    file_path: str = Field(..., description="Path of the processed file.", serialization_alias="file")
    chunks_ingested: int = Field(..., description="Number of chunks stored.", serialization_alias="ingested")