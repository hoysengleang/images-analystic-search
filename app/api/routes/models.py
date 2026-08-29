from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_embedding_manager
from app.embedding.manager import EmbeddingManager

router = APIRouter(tags=["models"])


class ModelMetadataResponse(BaseModel):
    provider: str
    model_name: str
    model_pretrained: str
    vector_size: int
    supports_text: bool = False
    is_default: bool = False


class ModelsResponse(BaseModel):
    models: list[ModelMetadataResponse]
    default_model: ModelMetadataResponse


@router.get("/models", response_model=ModelsResponse, summary="List embedding models")
def list_models(
    manager: EmbeddingManager = Depends(get_embedding_manager),
) -> ModelsResponse:
    default_metadata = manager.get_default_model_metadata()
    default_model = ModelMetadataResponse(
        **default_metadata.to_dict(),
        is_default=True,
    )

    models = [
        ModelMetadataResponse(
            **metadata.to_dict(),
            is_default=metadata == default_metadata,
        )
        for metadata in manager.list_available_models()
    ]

    return ModelsResponse(models=models, default_model=default_model)
