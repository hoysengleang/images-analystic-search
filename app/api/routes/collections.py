import json
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from pydantic import BaseModel

from app.core.errors import BadRequestError
from app.schemas.collection import (
    CollectionCreateRequest,
    CollectionResponse,
    CollectionStatsResponse,
)
from app.schemas.index import (
    IndexFolderRequest,
    IndexImagesRequest,
    IndexRequest,
    IndexResponse,
)
from app.schemas.image_record import ImageDeleteResponse, ImageRecordResponse
from app.schemas.search import (
    BatchSearchImagesRequest,
    BatchSearchResponse,
    SearchImagesRequest,
    SearchRequest,
    SearchResponse,
)
from app.services.collection_service import CollectionService, get_collection_service
from app.services.indexing_service import IndexingService, get_indexing_service
from app.services.search_service import SearchService, get_search_service
from app.services.vector_service import VectorService, get_vector_service

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


@router.get(
    "",
    response_model=CollectionsListResponse,
    summary="List collections",
)
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


@router.post(
    "/{collection_name}/index",
    response_model=IndexResponse,
    summary="Index images into collection",
)
def index_collection_images(
    collection_name: str,
    request: IndexImagesRequest,
    service: IndexingService = Depends(get_indexing_service),
) -> IndexResponse:
    return service.index(
        IndexRequest(
            collection_name=collection_name,
            images=request.images,
        )
    )


@router.post(
    "/{collection_name}/index/upload",
    response_model=IndexResponse,
    summary="Upload and index an image into collection",
)
async def upload_collection_image(
    collection_name: str,
    id: str = Form(...),
    image: UploadFile = File(...),
    metadata: Optional[str] = Form(default=None),
    service: IndexingService = Depends(get_indexing_service),
) -> IndexResponse:
    image_bytes = await image.read()
    return service.index_upload(
        collection_name=collection_name,
        image_id=id,
        image_bytes=image_bytes,
        filename=image.filename or id,
        content_type=image.content_type,
        metadata=_parse_upload_metadata(metadata),
    )


@router.post(
    "/{collection_name}/index/folder",
    response_model=IndexResponse,
    summary="Index images from a local folder",
)
def index_collection_folder(
    collection_name: str,
    request: IndexFolderRequest,
    service: IndexingService = Depends(get_indexing_service),
) -> IndexResponse:
    return service.index_folder(
        collection_name=collection_name,
        folder_path=request.folder_path,
        recursive=request.recursive,
        metadata=request.metadata,
    )


@router.post(
    "/{collection_name}/search",
    response_model=SearchResponse,
    summary="Search similar images in collection",
)
def search_collection_images(
    collection_name: str,
    request: SearchImagesRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return service.search(
        SearchRequest(
            collection_name=collection_name,
            source=request.source,
            top_k=request.top_k,
            min_score=request.min_score,
            filters=request.filters,
        )
    )


@router.post(
    "/{collection_name}/search/batch",
    response_model=BatchSearchResponse,
    summary="Search similar images with multiple query images",
)
def batch_search_collection_images(
    collection_name: str,
    request: BatchSearchImagesRequest,
    service: SearchService = Depends(get_search_service),
) -> BatchSearchResponse:
    return service.search_batch(
        collection_name=collection_name,
        sources=request.sources,
        mode=request.mode,
        top_k=request.top_k,
        min_score=request.min_score,
        filters=request.filters,
    )


@router.post(
    "/{collection_name}/search/upload",
    response_model=SearchResponse,
    summary="Upload an image and search similar images in collection",
)
async def upload_search_collection_images(
    collection_name: str,
    image: UploadFile = File(...),
    top_k: int = Form(default=10),
    min_score: Optional[float] = Form(default=None),
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    image_bytes = await image.read()
    return service.search_upload(
        collection_name=collection_name,
        image_bytes=image_bytes,
        filename=image.filename or "query",
        top_k=top_k,
        min_score=min_score,
    )


@router.get(
    "/{collection_name}/images/{image_id}",
    response_model=ImageRecordResponse,
    summary="Get image vector payload",
)
def get_collection_image(
    collection_name: str,
    image_id: str,
    service: VectorService = Depends(get_vector_service),
) -> ImageRecordResponse:
    return service.get_image(collection_name=collection_name, image_id=image_id)


@router.delete(
    "/{collection_name}/images/{image_id}",
    response_model=ImageDeleteResponse,
    summary="Delete image vector",
)
def delete_collection_image(
    collection_name: str,
    image_id: str,
    service: VectorService = Depends(get_vector_service),
) -> ImageDeleteResponse:
    return service.delete_image_record(
        collection_name=collection_name,
        image_id=image_id,
    )


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


def _parse_upload_metadata(metadata: Optional[str]) -> dict[str, object]:
    if metadata is None or not metadata.strip():
        return {}

    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise BadRequestError(
            message="Upload metadata must be a valid JSON object",
            code="INVALID_UPLOAD_METADATA",
            details={},
        ) from exc

    if not isinstance(parsed, dict):
        raise BadRequestError(
            message="Upload metadata must be a valid JSON object",
            code="INVALID_UPLOAD_METADATA",
            details={},
        )

    return parsed
