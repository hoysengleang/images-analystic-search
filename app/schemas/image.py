from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

SupportedImageSourceType = Literal["url", "path", "base64"]


class ImageSource(BaseModel):
    type: SupportedImageSourceType = Field(..., examples=["url"])
    value: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    display_image_url: Optional[str] = None

    @field_validator("type", mode="before")
    @classmethod
    def normalize_source_type(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("value", "display_image_url", mode="before")
    @classmethod
    def strip_optional_strings(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_source_value(self) -> "ImageSource":
        if self.type == "url" and not self.value.startswith(("http://", "https://")):
            raise ValueError("url image source value must start with http:// or https://")
        if self.type == "base64" and self.value.startswith("data:"):
            raise ValueError(
                "base64 image source value must not include a data URL prefix"
            )
        return self
