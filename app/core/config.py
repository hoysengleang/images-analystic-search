from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="OpenVisionSearch", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT", ge=1, le=65535)

    qdrant_url: str = Field(default="http://qdrant:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")

    default_embedding_provider: str = Field(
        default="openclip",
        alias="DEFAULT_EMBEDDING_PROVIDER",
    )
    default_model_name: str = Field(default="ViT-B-32", alias="DEFAULT_MODEL_NAME")
    default_model_pretrained: str = Field(
        default="laion2b_s34b_b79k",
        alias="DEFAULT_MODEL_PRETRAINED",
    )
    default_vector_size: int = Field(
        default=512,
        alias="DEFAULT_VECTOR_SIZE",
        ge=1,
    )

    allowed_image_root: Path = Field(
        default=Path("/data/images"),
        alias="ALLOWED_IMAGE_ROOT",
    )
    upload_dir: Path = Field(
        default=Path("/tmp/openvisionsearch/uploads"),
        alias="UPLOAD_DIR",
    )
    max_image_size_mb: int = Field(default=20, alias="MAX_IMAGE_SIZE_MB", ge=1)
    allowed_image_types: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp"],
        alias="ALLOWED_IMAGE_TYPES",
    )
    url_download_timeout_seconds: int = Field(
        default=10,
        alias="URL_DOWNLOAD_TIMEOUT_SECONDS",
        ge=1,
    )
    allow_private_urls: bool = Field(default=False, alias="ALLOW_PRIVATE_URLS")

    store_original_images: bool = Field(default=False, alias="STORE_ORIGINAL_IMAGES")
    storage_driver: str = Field(default="none", alias="STORAGE_DRIVER")

    default_top_k: int = Field(default=10, alias="DEFAULT_TOP_K", ge=1)
    max_top_k: int = Field(default=100, alias="MAX_TOP_K", ge=1)

    # Existing application metadata kept for the current FastAPI app factory.
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    api_prefix: str = Field(default="", alias="API_PREFIX")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("qdrant_api_key", mode="before")
    @classmethod
    def empty_string_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @field_validator(
        "app_env",
        "default_embedding_provider",
        "storage_driver",
        mode="before",
    )
    @classmethod
    def normalize_lowercase_strings(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("allowed_image_types")
    @classmethod
    def normalize_allowed_image_types(cls, value: list[str]) -> list[str]:
        return sorted({image_type.strip().lower() for image_type in value if image_type})

    @model_validator(mode="after")
    def validate_search_limits(self) -> "Settings":
        if self.default_top_k > self.max_top_k:
            raise ValueError("DEFAULT_TOP_K must be less than or equal to MAX_TOP_K")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
