from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.image import ImageSource

#: Every query image costs a model forward pass, so cap one request.
MAX_QUERY_IMAGES = 20

QUERY_DESCRIPTION = (
    "Plain-language description of what to find, for example "
    '"red running shoe with a white sole".'
)

TOP_K_DESCRIPTION = (
    "Number of results to return. Defaults to DEFAULT_TOP_K and is capped by MAX_TOP_K."
)


class SearchRequest(BaseModel):
    collection_name: str = Field(..., min_length=1, max_length=128, examples=["products"])
    source: ImageSource
    top_k: Optional[int] = Field(default=None, ge=1, description=TOP_K_DESCRIPTION)
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
    top_k: Optional[int] = Field(default=None, ge=1, description=TOP_K_DESCRIPTION)
    min_score: Optional[float] = Field(default=None)
    filters: dict[str, Any] = Field(default_factory=dict)


class TextSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description=QUERY_DESCRIPTION,
        examples=["red running shoe"],
    )
    top_k: Optional[int] = Field(default=None, ge=1, description=TOP_K_DESCRIPTION)
    min_score: Optional[float] = Field(
        default=None,
        description=(
            "Text-to-image scores are much lower than image-to-image scores. "
            "Start around 0.2 rather than reusing an image threshold."
        ),
    )
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class HybridSearchRequest(BaseModel):
    """An image query nudged by words, for example "this shoe, but blue"."""

    source: ImageSource
    query: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description=QUERY_DESCRIPTION,
        examples=["but in blue"],
    )
    text_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "How much the text pulls the query away from the image. "
            "0.0 is a pure image search, 1.0 is a pure text search."
        ),
    )
    top_k: Optional[int] = Field(default=None, ge=1, description=TOP_K_DESCRIPTION)
    min_score: Optional[float] = Field(default=None)
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class BatchSearchImagesRequest(BaseModel):
    sources: list[ImageSource] = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_IMAGES,
    )
    mode: Literal["average", "separate"] = "average"
    top_k: Optional[int] = Field(default=None, ge=1, description=TOP_K_DESCRIPTION)
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
