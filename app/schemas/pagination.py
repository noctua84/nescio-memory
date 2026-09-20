from typing import Generic, TypeVar, List

from pydantic import BaseModel, Field

T = TypeVar("T")

class PaginationResponse(BaseModel, Generic[T]):
    data: List[T] = Field(..., serialization_alias="data")
    total_count: int = Field(serialization_alias="totalCount")
    page: int = Field(serialization_alias="page")
    page_size: int = Field(serialization_alias="pageSize")