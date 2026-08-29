"""Request and response schema package."""

from app.schemas.collection import (
    CollectionCreateRequest,
    CollectionModelConfig,
    CollectionResponse,
    CollectionStatsResponse,
)
from app.schemas.common import ErrorResponse
from app.schemas.image import ImageSource
from app.schemas.image_record import (
    ImageDeleteResponse,
    ImageListResponse,
    ImageRecordResponse,
)
from app.schemas.index import (
    IndexFolderRequest,
    IndexImageItem,
    IndexImagesRequest,
    IndexRequest,
    IndexResponse,
)
from app.schemas.search import (
    BatchSearchGroup,
    BatchSearchImagesRequest,
    BatchSearchResponse,
    HybridSearchRequest,
    SearchImagesRequest,
    SearchRequest,
    SearchResponse,
    SearchResult,
    TextSearchRequest,
)

__all__ = [
    "CollectionCreateRequest",
    "CollectionModelConfig",
    "CollectionResponse",
    "CollectionStatsResponse",
    "ErrorResponse",
    "ImageSource",
    "ImageDeleteResponse",
    "ImageListResponse",
    "ImageRecordResponse",
    "IndexFolderRequest",
    "IndexImageItem",
    "IndexImagesRequest",
    "IndexRequest",
    "IndexResponse",
    "SearchRequest",
    "SearchImagesRequest",
    "BatchSearchGroup",
    "BatchSearchImagesRequest",
    "BatchSearchResponse",
    "HybridSearchRequest",
    "SearchResponse",
    "SearchResult",
    "TextSearchRequest",
]
