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
    created_at: Optional[str] = None


class ImageDeleteResponse(BaseModel):
    id: str
    collection_name: str
    deleted: bool = True
