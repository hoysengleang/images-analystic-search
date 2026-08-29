"""Composition root for the FastAPI application.

Service classes take their collaborators explicitly and never build them at
import time. All wiring lives here, so routes have one place to depend on and
tests have one place to override.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.core.errors import ServiceUnavailableError
from app.embedding.manager import EmbeddingManager
from app.embedding.registry import get_embedding_registry
from app.providers.qdrant_provider import get_qdrant_client
from app.services.collection_metadata_service import CollectionMetadataService
from app.services.collection_service import CollectionService
from app.services.image_loader import ImageLoader
from app.services.indexing_service import IndexingService
from app.services.search_service import SearchService
from app.services.vector_service import VectorService

__all__ = [
    "get_collection_metadata_service",
    "get_collection_service",
    "get_embedding_manager",
    "get_image_loader",
    "get_indexing_service",
    "get_qdrant_client",
    "get_search_service",
    "get_vector_service",
]


@lru_cache
def get_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager(
        settings=get_settings(),
        registry=get_embedding_registry(),
    )


@lru_cache
def get_collection_metadata_service() -> CollectionMetadataService:
    settings: Settings = get_settings()
    return CollectionMetadataService(settings.collection_metadata_path)


@lru_cache
def get_image_loader() -> ImageLoader:
    return ImageLoader(settings=get_settings())


#: Backends that can serve searches, keyed by the SEARCH_ENGINE value.
#: Milestone 1 registers the native engine alongside Qdrant here.
SEARCH_ENGINE_BUILDERS = {
    "qdrant": lambda: VectorService(qdrant_client=get_qdrant_client()),
}


@lru_cache
def get_vector_service() -> VectorService:
    engine = get_settings().search_engine
    try:
        build = SEARCH_ENGINE_BUILDERS[engine]
    except KeyError as exc:
        raise ServiceUnavailableError(
            message=f"No search engine is registered for SEARCH_ENGINE={engine}",
            code="UNKNOWN_SEARCH_ENGINE",
            details={"engine": engine, "available": sorted(SEARCH_ENGINE_BUILDERS)},
        ) from exc

    return build()


@lru_cache
def get_collection_service() -> CollectionService:
    return CollectionService(
        settings=get_settings(),
        qdrant_client=get_qdrant_client(),
        metadata_service=get_collection_metadata_service(),
        embedding_manager=get_embedding_manager(),
    )


@lru_cache
def get_indexing_service() -> IndexingService:
    return IndexingService(
        settings=get_settings(),
        metadata_service=get_collection_metadata_service(),
        image_loader=get_image_loader(),
        embedding_manager=get_embedding_manager(),
        vector_service=get_vector_service(),
    )


@lru_cache
def get_search_service() -> SearchService:
    return SearchService(
        settings=get_settings(),
        metadata_service=get_collection_metadata_service(),
        image_loader=get_image_loader(),
        embedding_manager=get_embedding_manager(),
        vector_service=get_vector_service(),
    )
