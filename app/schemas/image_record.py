from typing import Any, Optional

from pydantic import BaseModel, Field


class ImageRecordResponse(BaseModel):
    id: str
    collection_name: str
    source_type: Optional[str] = None
    source_value: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_revision: Optional[str] = None
    vector_dimension: Optional[int] = None
    created_at: Optional[str] = None


class ImageListResponse(BaseModel):
    collection_name: str
    images: list[ImageRecordResponse] = Field(default_factory=list)
    next_cursor: Optional[str] = Field(
        default=None,
        description="Pass back as `cursor` to fetch the next page.",
    )


class ImageDeleteResponse(BaseModel):
    id: str
    collection_name: str
    deleted: bool = True
