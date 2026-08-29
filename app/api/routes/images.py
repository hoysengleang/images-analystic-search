"""Read and remove individual indexed images."""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.dependencies import get_vector_service
from app.schemas.image_record import (
    ImageDeleteResponse,
    ImageListResponse,
    ImageRecordResponse,
)
from app.services.vector_service import VectorService

router = APIRouter(prefix="/collections/{collection_name}/images", tags=["images"])


@router.get("", response_model=ImageListResponse, summary="List indexed images")
def list_images(
    collection_name: str,
    limit: int = Query(default=50, ge=1, le=1000),
    cursor: Optional[str] = Query(
        default=None,
        description="`next_cursor` value from a previous page.",
    ),
    service: VectorService = Depends(get_vector_service),
) -> ImageListResponse:
    return service.list_images(
        collection_name=collection_name,
        limit=limit,
        cursor=cursor,
    )


@router.get(
    "/{image_id}",
    response_model=ImageRecordResponse,
    summary="Get one indexed image",
)
def get_image(
    collection_name: str,
    image_id: str,
    service: VectorService = Depends(get_vector_service),
) -> ImageRecordResponse:
    return service.get_image(collection_name=collection_name, image_id=image_id)


@router.delete(
    "/{image_id}",
    response_model=ImageDeleteResponse,
    summary="Delete one indexed image",
)
def delete_image(
    collection_name: str,
    image_id: str,
    service: VectorService = Depends(get_vector_service),
) -> ImageDeleteResponse:
    return service.delete_image_record(
        collection_name=collection_name,
        image_id=image_id,
    )
