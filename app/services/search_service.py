from __future__ import annotations

from typing import Optional

from app.core.config import Settings, get_settings
from app.core.errors import BadRequestError
from app.embedding.manager import EmbeddingManager, get_embedding_manager
from app.schemas.image import ImageSource
from app.schemas.search import (
    BatchSearchGroup,
    BatchSearchResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.collection_metadata_service import (
    CollectionMetadataService,
    get_collection_metadata_service,
)
from app.services.image_loader import ImageLoader, get_image_loader
from app.services.vector_service import VectorService, get_vector_service
from app.utils.image_utils import validate_and_load_image


class SearchService:
    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        metadata_service: Optional[CollectionMetadataService] = None,
        image_loader: Optional[ImageLoader] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        vector_service: Optional[VectorService] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.metadata_service = metadata_service or get_collection_metadata_service()
        self.image_loader = image_loader or get_image_loader()
        self.embedding_manager = embedding_manager or get_embedding_manager()
        self.vector_service = vector_service or get_vector_service()

    def search(self, request: SearchRequest) -> SearchResponse:
        if request.top_k > self.settings.max_top_k:
            raise BadRequestError(
                message="top_k exceeds the configured maximum",
                code="TOP_K_TOO_LARGE",
                details={
                    "top_k": request.top_k,
                    "max_top_k": self.settings.max_top_k,
                },
            )

        collection_metadata = self.metadata_service.get(request.collection_name)
        embedding_provider = self.embedding_manager.get_provider(
            provider_name=collection_metadata.embedding_provider,
            model_name=collection_metadata.embedding_model,
            model_pretrained=collection_metadata.embedding_pretrained,
            vector_size=collection_metadata.vector_size,
        )

        image = self.image_loader.load_from_source(request.source)
        vector = embedding_provider.embed_image(image)
        results = self.vector_service.search(
            collection_name=collection_metadata.collection_name,
            vector=vector,
            top_k=request.top_k,
            filters=request.filters,
            min_score=request.min_score,
        )

        return SearchResponse(
            collection_name=collection_metadata.collection_name,
            top_k=request.top_k,
            results=results,
        )

    def search_upload(
        self,
        *,
        collection_name: str,
        image_bytes: bytes,
        filename: str,
        top_k: int,
        min_score: Optional[float] = None,
    ) -> SearchResponse:
        if top_k > self.settings.max_top_k:
            raise BadRequestError(
                message="top_k exceeds the configured maximum",
                code="TOP_K_TOO_LARGE",
                details={
                    "top_k": top_k,
                    "max_top_k": self.settings.max_top_k,
                },
            )

        collection_metadata = self.metadata_service.get(collection_name)
        embedding_provider = self.embedding_manager.get_provider(
            provider_name=collection_metadata.embedding_provider,
            model_name=collection_metadata.embedding_model,
            model_pretrained=collection_metadata.embedding_pretrained,
            vector_size=collection_metadata.vector_size,
        )

        validated_image = validate_and_load_image(
            image_bytes,
            max_size_mb=self.settings.max_image_size_mb,
            extension=filename,
        )
        vector = embedding_provider.embed_image(validated_image.image)
        results = self.vector_service.search(
            collection_name=collection_metadata.collection_name,
            vector=vector,
            top_k=top_k,
            min_score=min_score,
        )

        return SearchResponse(
            collection_name=collection_metadata.collection_name,
            top_k=top_k,
            results=results,
        )

    def search_batch(
        self,
        *,
        collection_name: str,
        sources: list[ImageSource],
        mode: str,
        top_k: int,
        min_score: Optional[float] = None,
        filters: Optional[dict[str, object]] = None,
    ) -> BatchSearchResponse:
        if top_k > self.settings.max_top_k:
            raise BadRequestError(
                message="top_k exceeds the configured maximum",
                code="TOP_K_TOO_LARGE",
                details={
                    "top_k": top_k,
                    "max_top_k": self.settings.max_top_k,
                },
            )

        collection_metadata = self.metadata_service.get(collection_name)
        embedding_provider = self.embedding_manager.get_provider(
            provider_name=collection_metadata.embedding_provider,
            model_name=collection_metadata.embedding_model,
            model_pretrained=collection_metadata.embedding_pretrained,
            vector_size=collection_metadata.vector_size,
        )
        vectors = [
            embedding_provider.embed_image(self.image_loader.load_from_source(source))
            for source in sources
        ]

        if mode == "average":
            average_vector = self._normalize_vector(self._average_vectors(vectors))
            results = self.vector_service.search(
                collection_name=collection_metadata.collection_name,
                vector=average_vector,
                top_k=top_k,
                filters=filters or {},
                min_score=min_score,
            )
            return BatchSearchResponse(
                collection_name=collection_metadata.collection_name,
                mode="average",
                top_k=top_k,
                results=results,
                groups=[],
            )

        groups = [
            BatchSearchGroup(
                source_index=index,
                source=source,
                results=self.vector_service.search(
                    collection_name=collection_metadata.collection_name,
                    vector=vector,
                    top_k=top_k,
                    filters=filters or {},
                    min_score=min_score,
                ),
            )
            for index, (source, vector) in enumerate(zip(sources, vectors))
        ]
        return BatchSearchResponse(
            collection_name=collection_metadata.collection_name,
            mode="separate",
            top_k=top_k,
            results=[],
            groups=groups,
        )

    def _average_vectors(self, vectors: list[list[float]]) -> list[float]:
        vector_count = len(vectors)
        dimension = len(vectors[0])
        return [
            sum(vector[index] for vector in vectors) / vector_count
            for index in range(dimension)
        ]

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            raise BadRequestError(
                message="Average query vector cannot be zero",
                code="ZERO_QUERY_VECTOR",
                details={},
            )
        return [value / norm for value in vector]


def get_search_service() -> SearchService:
    return SearchService()
