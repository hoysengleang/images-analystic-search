"""Response models for service metadata endpoints."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    app_name: str
    status: str
    search_engine: str
    qdrant: Optional[dict[str, Any]] = None


class ModelMetadataResponse(BaseModel):
    provider: str
    model_name: str
    model_pretrained: str
    vector_size: int = Field(..., ge=1)
    supports_text: bool = False
    is_default: bool = False


class ModelsResponse(BaseModel):
    models: list[ModelMetadataResponse] = Field(default_factory=list)
    default_model: ModelMetadataResponse
