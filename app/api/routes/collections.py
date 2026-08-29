"""Collection lifecycle endpoints."""

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from app.dependencies import get_collection_service
from app.schemas.collection import (
    CollectionCreateRequest,
    CollectionResponse,
    CollectionStatsResponse,
)
from app.services.collection_service import CollectionService

router = APIRouter(prefix="/collections", tags=["collections"])


class CollectionsListResponse(BaseModel):
    collections: list[CollectionResponse]


@router.post(
    "",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create collection",
)
def create_collection(
    request: CollectionCreateRequest,
    service: CollectionService = Depends(get_collection_service),
) -> CollectionResponse:
    return service.create_collection(request)


@router.get("", response_model=CollectionsListResponse, summary="List collections")
def list_collections(
    service: CollectionService = Depends(get_collection_service),
) -> CollectionsListResponse:
    return CollectionsListResponse(collections=service.list_collections())


@router.get(
    "/{collection_name}/stats",
    response_model=CollectionStatsResponse,
    summary="Get collection stats",
)
def get_collection_stats(
    collection_name: str,
    service: CollectionService = Depends(get_collection_service),
) -> CollectionStatsResponse:
    return service.get_collection_stats(collection_name)


@router.delete(
    "/{collection_name}",
    response_model=CollectionResponse,
    summary="Delete collection",
)
def delete_collection(
    collection_name: str,
    service: CollectionService = Depends(get_collection_service),
) -> CollectionResponse:
    return service.delete_collection(collection_name)
