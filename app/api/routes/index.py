"""Endpoints that turn images into stored vectors."""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.form_parsing import parse_metadata_form_field, read_upload_within_limit
from app.core.config import Settings, get_settings
from app.dependencies import get_indexing_service
from app.schemas.index import (
    IndexFolderRequest,
    IndexImagesRequest,
    IndexRequest,
    IndexResponse,
)
from app.services.indexing_service import IndexingService

router = APIRouter(prefix="/collections/{collection_name}", tags=["indexing"])


@router.post("/index", response_model=IndexResponse, summary="Index images")
def index_images(
    collection_name: str,
    request: IndexImagesRequest,
    service: IndexingService = Depends(get_indexing_service),
) -> IndexResponse:
    return service.index(
        IndexRequest(collection_name=collection_name, images=request.images)
    )


@router.post(
    "/index/upload",
    response_model=IndexResponse,
    summary="Upload and index an image",
)
async def index_uploaded_image(
    collection_name: str,
    id: str = Form(..., description="Stable image id used for later lookups."),
    image: UploadFile = File(...),
    metadata: Optional[str] = Form(
        default=None,
        description="Optional JSON object stored alongside the vector.",
    ),
    service: IndexingService = Depends(get_indexing_service),
    settings: Settings = Depends(get_settings),
) -> IndexResponse:
    image_bytes = await read_upload_within_limit(
        image,
        max_bytes=settings.max_image_size_bytes,
    )
    return service.index_upload(
        collection_name=collection_name,
        image_id=id,
        image_bytes=image_bytes,
        filename=image.filename or id,
        content_type=image.content_type,
        metadata=parse_metadata_form_field(metadata),
    )


@router.post(
    "/index/folder",
    response_model=IndexResponse,
    summary="Index every image in a mounted folder",
)
def index_folder(
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
