from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.image import ImageSource


class SearchRequest(BaseModel):
    collection_name: str = Field(..., min_length=1, max_length=128, examples=["products"])
    source: ImageSource
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: Optional[float] = Field(default=None)
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("collection_name", mode="before")
    @classmethod
    def normalize_collection_name(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class SearchImagesRequest(BaseModel):
    source: ImageSource
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: Optional[float] = Field(default=None)
    filters: dict[str, Any] = Field(default_factory=dict)


class BatchSearchImagesRequest(BaseModel):
    sources: list[ImageSource] = Field(..., min_length=1)
    mode: Literal["average", "separate"] = "average"
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: Optional[float] = Field(default=None)
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class SearchResult(BaseModel):
    id: str
    score: float
    source_type: Optional[str] = None
    source_value: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    display_image_url: Optional[str] = None


class SearchResponse(BaseModel):
    collection_name: str
    top_k: int
    results: list[SearchResult] = Field(default_factory=list)


class BatchSearchGroup(BaseModel):
    source_index: int
    source: ImageSource
    results: list[SearchResult] = Field(default_factory=list)


class BatchSearchResponse(BaseModel):
    collection_name: str
    mode: Literal["average", "separate"]
    top_k: int
    results: list[SearchResult] = Field(default_factory=list)
    groups: list[BatchSearchGroup] = Field(default_factory=list)
