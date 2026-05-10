from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.image import ImageSource


class IndexImageItem(BaseModel):
    id: str = Field(..., min_length=1, max_length=256, examples=["product_001"])
    source: ImageSource
    metadata: dict[str, Any] = Field(default_factory=dict)
    display_image_url: Optional[str] = None

    @field_validator("id", "display_image_url", mode="before")
    @classmethod
    def strip_optional_strings(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class IndexRequest(BaseModel):
    collection_name: str = Field(..., min_length=1, max_length=128, examples=["products"])
    images: list[IndexImageItem] = Field(..., min_length=1)

    @field_validator("collection_name", mode="before")
    @classmethod
    def normalize_collection_name(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class IndexResponse(BaseModel):
    collection_name: str
    indexed_count: int = Field(..., ge=0)
    failed_count: int = Field(default=0, ge=0)
    ids: list[str] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


class IndexImagesRequest(BaseModel):
    images: list[IndexImageItem] = Field(..., min_length=1)


class IndexFolderRequest(BaseModel):
    folder_path: str = Field(..., min_length=1, examples=["/data/images/products"])
    recursive: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("folder_path", mode="before")
    @classmethod
    def normalize_folder_path(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value
