from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.constants import (
    SUPPORTED_DISTANCE_METRICS,
    SUPPORTED_IMAGE_FRAMINGS,
)


class CollectionModelConfig(BaseModel):
    provider: str = Field(..., min_length=1, examples=["openclip"])
    name: str = Field(..., min_length=1, examples=["ViT-B-32"])
    pretrained: str = Field(..., min_length=1, examples=["laion2b_s34b_b79k"])
    vector_size: int = Field(..., ge=1, examples=[512])
    distance: str = Field(default="cosine", min_length=1, examples=["cosine"])
    framing: str = Field(
        default="pad",
        examples=["pad"],
        description=(
            "How a non-square image is fitted to the model input. Pinned per "
            "collection: vectors framed differently are not comparable."
        ),
    )
    views: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Views averaged per image. Also pinned per collection.",
    )

    @field_validator("framing")
    @classmethod
    def validate_framing(cls, value: str) -> str:
        if value not in SUPPORTED_IMAGE_FRAMINGS:
            raise ValueError(
                f"framing must be one of: {', '.join(sorted(SUPPORTED_IMAGE_FRAMINGS))}"
            )
        return value

    @field_validator("provider", "distance", "framing", mode="before")
    @classmethod
    def normalize_lowercase_fields(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("name", "pretrained", mode="before")
    @classmethod
    def strip_string_fields(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("distance")
    @classmethod
    def validate_distance(cls, value: str) -> str:
        if value not in SUPPORTED_DISTANCE_METRICS:
            raise ValueError(
                "distance must be one of: "
                f"{', '.join(sorted(SUPPORTED_DISTANCE_METRICS))}"
            )
        return value


class CollectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, examples=["products"])
    model: Optional[CollectionModelConfig] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class CollectionResponse(BaseModel):
    name: str
    model: CollectionModelConfig
    points_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class CollectionStatsResponse(BaseModel):
    name: str
    points_count: int = Field(default=0, ge=0)
    vector_size: int = Field(..., ge=1)
    distance: str
    model: CollectionModelConfig

    model_config = ConfigDict(from_attributes=True)


class CollectionsListResponse(BaseModel):
    collections: list[CollectionResponse] = Field(default_factory=list)
