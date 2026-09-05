from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import (
    MIME_TYPE_TO_EXTENSION,
    SUPPORTED_DISTANCE_METRICS,
    SUPPORTED_IMAGE_FRAMINGS,
)


class Settings(BaseSettings):
    """Environment-driven application settings.

    Every field is read once at startup and cached by :func:`get_settings`.
    """

    # --- Application ---------------------------------------------------
    app_name: str = Field(default="OpenVisionSearch", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT", ge=1, le=65535)
    api_prefix: str = Field(default="", alias="API_PREFIX")
    app_data_dir: Path = Field(default=Path("data"), alias="APP_DATA_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")

    # --- Access control ------------------------------------------------
    api_key: Optional[str] = Field(default=None, alias="API_KEY")
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        alias="CORS_ALLOW_ORIGINS",
    )

    # --- Vector database -----------------------------------------------
    # The native engine lands in Milestone 1; until then Qdrant is the only
    # backend, so defaulting to "native" would promise something that is not
    # wired up and fail confusingly at the first search.
    search_engine: str = Field(default="qdrant", alias="SEARCH_ENGINE")
    qdrant_url: str = Field(default="http://qdrant:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    collection_metadata_path: Path = Field(
        default=Path("data/collections.json"),
        alias="COLLECTION_METADATA_PATH",
    )

    # --- Embedding model -----------------------------------------------
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
    default_distance: str = Field(default="cosine", alias="DEFAULT_DISTANCE")
    # "pad" keeps the whole frame; the stock CLIP transform centre-crops and
    # discards ~40% of a portrait photo. Measured never worse, much better on
    # non-square images, so it is the default.
    image_framing: str = Field(default="pad", alias="IMAGE_FRAMING")
    # Views averaged per image. 1 is fastest; 3 buys accuracy for ~2x the work.
    embed_views: int = Field(default=1, alias="EMBED_VIEWS", ge=1, le=3)
    embed_batch_size: int = Field(
        default=16,
        alias="EMBED_BATCH_SIZE",
        ge=1,
        le=256,
    )
    warmup_model_on_startup: bool = Field(
        default=False,
        alias="WARMUP_MODEL_ON_STARTUP",
    )

    # --- ONNX Runtime provider -----------------------------------------
    # Read only when a collection uses the "onnx" provider. Build the files
    # with scripts/export_onnx.py. The text encoder is optional: without it
    # the provider serves image search only and refuses text queries.
    onnx_image_model_path: Optional[str] = Field(
        default=None,
        alias="ONNX_IMAGE_MODEL_PATH",
    )
    onnx_text_model_path: Optional[str] = Field(
        default=None,
        alias="ONNX_TEXT_MODEL_PATH",
    )
    onnx_tokenizer_path: Optional[str] = Field(
        default=None,
        alias="ONNX_TOKENIZER_PATH",
    )

    # --- Query-side object detection -----------------------------------
    # Detection narrows a query image before it is embedded; it never touches
    # stored vectors, so it is a server-wide switch rather than something
    # pinned per collection, and turning it on needs no reindex.
    detector_provider: str = Field(default="none", alias="DETECTOR_PROVIDER")
    detector_model_path: Optional[str] = Field(
        default=None,
        alias="DETECTOR_MODEL_PATH",
    )
    detector_tokenizer_path: Optional[str] = Field(
        default=None,
        alias="DETECTOR_TOKENIZER_PATH",
    )
    # Open-vocabulary detectors have no class list, so they need to be told what
    # to look for. This is the fallback when a caller does not say.
    #
    # The phrasing matters more than it looks: OWL-ViT scores against CLIP text
    # embeddings, so a prompt shaped like a caption beats a bare noun. Measured
    # on one cluttered scene, "a photo of an object" peaked at 0.16 while "an
    # object" found nothing at all. A caller who names the thing does far better
    # still - "a handbag" scored 0.73 on the same picture - which is why
    # detect_prompt exists.
    detector_prompts: list[str] = Field(
        default_factory=lambda: ["a photo of an object"],
        alias="DETECTOR_PROMPTS",
    )
    # Low, because generic prompts score low. Raise it if you see regions that
    # are not really there.
    detector_min_score: float = Field(
        default=0.05,
        alias="DETECTOR_MIN_SCORE",
        ge=0.0,
        le=1.0,
    )
    # One by default, and the default matters more than it looks. Folding
    # regions by their best score can only raise a product's score, never lower
    # it, so a weak extra region lifts whatever it happens to resemble. Measured
    # on a bag photographed beside a laptop: searching the strongest region
    # alone put the bag 0.257 clear of the runner-up, while adding a second,
    # 0.09-confidence region lifted the laptop and cut the lead to 0.073 -
    # worse than not detecting at all. Raise this only for "find everything in
    # this photo" behaviour, not to find one product.
    max_query_regions: int = Field(
        default=1,
        alias="MAX_QUERY_REGIONS",
        ge=1,
        le=10,
    )

    # --- Image sources -------------------------------------------------
    allowed_image_root: Path = Field(
        default=Path("/data/images"),
        alias="ALLOWED_IMAGE_ROOT",
    )
    max_image_size_mb: int = Field(default=20, alias="MAX_IMAGE_SIZE_MB", ge=1)
    max_request_body_mb: int = Field(
        default=25,
        alias="MAX_REQUEST_BODY_MB",
        ge=1,
    )
    max_image_pixels: int = Field(
        default=40_000_000,
        alias="MAX_IMAGE_PIXELS",
        ge=1,
        description="Decoded width x height ceiling, guarding decompression bombs.",
    )
    allowed_image_types: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp"],
        alias="ALLOWED_IMAGE_TYPES",
    )
    url_download_timeout_seconds: int = Field(
        default=10,
        alias="URL_DOWNLOAD_TIMEOUT_SECONDS",
        ge=1,
    )
    max_redirects: int = Field(default=3, alias="MAX_REDIRECTS", ge=0, le=10)
    allow_private_urls: bool = Field(default=False, alias="ALLOW_PRIVATE_URLS")

    # --- Search limits -------------------------------------------------
    default_top_k: int = Field(default=10, alias="DEFAULT_TOP_K", ge=1)
    max_top_k: int = Field(default=100, alias="MAX_TOP_K", ge=1)
    # Section 13 of the specification asks for more candidates than the final
    # limit, because results are collapsed after retrieval: several query
    # regions can land on the same image, and one product has several images.
    search_candidate_multiplier: int = Field(
        default=4,
        alias="SEARCH_CANDIDATE_MULTIPLIER",
        ge=1,
        le=20,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def allowed_image_extensions(self) -> frozenset:
        """Canonical file extensions derived from ``ALLOWED_IMAGE_TYPES``."""
        return frozenset(
            MIME_TYPE_TO_EXTENSION[mime_type]
            for mime_type in self.allowed_image_types
            if mime_type in MIME_TYPE_TO_EXTENSION
        )

    @property
    def database_path(self) -> Path:
        """SQLite lives under the data directory unless overridden."""
        return self.app_data_dir / "app.db"

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    @property
    def max_request_body_bytes(self) -> int:
        return self.max_request_body_mb * 1024 * 1024

    @field_validator(
        "api_key",
        "qdrant_api_key",
        "onnx_image_model_path",
        "onnx_text_model_path",
        "onnx_tokenizer_path",
        "detector_model_path",
        "detector_tokenizer_path",
        mode="before",
    )
    @classmethod
    def empty_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("image_framing")
    @classmethod
    def validate_image_framing(cls, value: str) -> str:
        if value not in SUPPORTED_IMAGE_FRAMINGS:
            raise ValueError(
                "IMAGE_FRAMING must be one of: "
                f"{', '.join(sorted(SUPPORTED_IMAGE_FRAMINGS))}"
            )
        return value

    @field_validator("search_engine")
    @classmethod
    def validate_search_engine(cls, value: str) -> str:
        if value == "native":
            raise ValueError(
                "SEARCH_ENGINE=native is not implemented yet; the native "
                "engine arrives in Milestone 1. Use SEARCH_ENGINE=qdrant."
            )
        if value != "qdrant":
            raise ValueError("SEARCH_ENGINE must be 'qdrant'")
        return value

    @field_validator(
        "app_env",
        "default_embedding_provider",
        "detector_provider",
        "default_distance",
        "search_engine",
        "image_framing",
        mode="before",
    )
    @classmethod
    def normalize_lowercase_strings(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("allowed_image_types")
    @classmethod
    def normalize_allowed_image_types(cls, value: list) -> list:
        normalized = sorted(
            {image_type.strip().lower() for image_type in value if image_type}
        )
        unsupported = [
            image_type
            for image_type in normalized
            if image_type not in MIME_TYPE_TO_EXTENSION
        ]
        if unsupported:
            supported = ", ".join(sorted(MIME_TYPE_TO_EXTENSION))
            raise ValueError(
                f"ALLOWED_IMAGE_TYPES contains unsupported types: "
                f"{', '.join(unsupported)}. Supported types are: {supported}"
            )
        if not normalized:
            raise ValueError("ALLOWED_IMAGE_TYPES must contain at least one MIME type")
        return normalized

    @field_validator("cors_allow_origins")
    @classmethod
    def normalize_cors_origins(cls, value: list) -> list:
        return [origin.strip() for origin in value if origin and origin.strip()]

    @field_validator("default_distance")
    @classmethod
    def validate_default_distance(cls, value: str) -> str:
        if value not in SUPPORTED_DISTANCE_METRICS:
            raise ValueError(
                "DEFAULT_DISTANCE must be one of: "
                f"{', '.join(sorted(SUPPORTED_DISTANCE_METRICS))}"
            )
        return value

    @model_validator(mode="after")
    def validate_search_limits(self) -> "Settings":
        if self.default_top_k > self.max_top_k:
            raise ValueError("DEFAULT_TOP_K must be less than or equal to MAX_TOP_K")
        return self

    @model_validator(mode="after")
    def validate_onnx_paths(self) -> "Settings":
        """A misconfigured default provider should fail at boot, not at search."""
        if self.default_embedding_provider == "onnx" and not self.onnx_image_model_path:
            raise ValueError(
                "DEFAULT_EMBEDDING_PROVIDER=onnx requires ONNX_IMAGE_MODEL_PATH; "
                "export one with scripts/export_onnx.py"
            )
        if self.onnx_text_model_path and not self.onnx_tokenizer_path:
            raise ValueError(
                "ONNX_TEXT_MODEL_PATH requires ONNX_TOKENIZER_PATH; the text "
                "encoder cannot run without the tokenizer it was exported with"
            )
        return self

    @property
    def detection_enabled(self) -> bool:
        """Whether a caller may ask the server to find the objects itself."""
        return self.detector_provider != "none"

    @model_validator(mode="after")
    def validate_detector_paths(self) -> "Settings":
        """A detector named but not supplied should fail at boot, not at search."""
        if self.detection_enabled and not (
            self.detector_model_path and self.detector_tokenizer_path
        ):
            raise ValueError(
                "DETECTOR_PROVIDER needs DETECTOR_MODEL_PATH and "
                "DETECTOR_TOKENIZER_PATH; export them with "
                "scripts/export_detector_onnx.py, or set DETECTOR_PROVIDER=none"
            )
        if not self.detector_prompts:
            raise ValueError(
                "DETECTOR_PROMPTS must name at least one thing to look for; an "
                "open-vocabulary detector has no class list of its own"
            )
        return self

    @model_validator(mode="after")
    def validate_body_limit(self) -> "Settings":
        # A body cap below the image cap would reject uploads the image rules
        # accept, which is impossible to debug from the outside.
        if self.max_request_body_mb < self.max_image_size_mb:
            raise ValueError(
                "MAX_REQUEST_BODY_MB must be greater than or equal to MAX_IMAGE_SIZE_MB"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
