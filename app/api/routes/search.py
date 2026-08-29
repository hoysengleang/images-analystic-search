from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.form_parsing import read_upload_within_limit
from app.core.config import Settings, get_settings
from app.dependencies import get_search_service
from app.schemas.search import (
    BatchSearchImagesRequest,
    BatchSearchResponse,
    HybridSearchRequest,
    SearchImagesRequest,
    SearchRequest,
    SearchResponse,
    TextSearchRequest,
)
from app.services.search_service import SearchService

router = APIRouter(prefix="/collections/{collection_name}", tags=["search"])


@router.post("/search", response_model=SearchResponse, summary="Search similar images")
def search_images(
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
    "/search/text",
    response_model=SearchResponse,
    summary="Search images with a text description",
)
def search_by_text(
    collection_name: str,
    request: TextSearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return service.search_text(
        collection_name=collection_name,
        query=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
        filters=request.filters,
    )


@router.post(
    "/search/hybrid",
    response_model=SearchResponse,
    summary="Search with an image steered by a text description",
)
def search_hybrid(
    collection_name: str,
    request: HybridSearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return service.search_hybrid(
        collection_name=collection_name,
        source=request.source,
        query=request.query,
        text_weight=request.text_weight,
        top_k=request.top_k,
        min_score=request.min_score,
        filters=request.filters,
    )


@router.post(
    "/search/upload",
    response_model=SearchResponse,
    summary="Upload an image and search similar images",
)
async def search_uploaded_image(
    collection_name: str,
    image: UploadFile = File(...),
    top_k: Optional[int] = Form(default=None),
    min_score: Optional[float] = Form(default=None),
    service: SearchService = Depends(get_search_service),
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    image_bytes = await read_upload_within_limit(
        image,
        max_bytes=settings.max_image_size_bytes,
    )
    return service.search_upload(
        collection_name=collection_name,
        image_bytes=image_bytes,
        filename=image.filename or "query",
        top_k=top_k,
        min_score=min_score,
    )


@router.post(
    "/search/batch",
    response_model=BatchSearchResponse,
    summary="Search with several query images at once",
)
def search_images_batch(
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
