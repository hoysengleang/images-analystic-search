from __future__ import annotations

from typing import Any, Optional

from app.core.config import Settings
from app.core.errors import BadRequestError
from app.embedding.base import EmbeddingProvider
from app.embedding.manager import EmbeddingManager, get_configured_provider
from app.schemas.image import ImageSource
from app.schemas.search import (
    BatchSearchGroup,
    BatchSearchResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.collection_metadata_service import (
    CollectionMetadata,
    CollectionMetadataService,
)
from app.services.image_loader import ImageLoader
from app.services.vector_service import VectorService
from app.utils.image_utils import validate_and_load_image


class SearchService:
    def __init__(
        self,
        *,
        settings: Settings,
        metadata_service: CollectionMetadataService,
        image_loader: ImageLoader,
        embedding_manager: EmbeddingManager,
        vector_service: VectorService,
    ) -> None:
        self.settings = settings
        self.metadata_service = metadata_service
        self.image_loader = image_loader
        self.embedding_manager = embedding_manager
        self.vector_service = vector_service

    def search(self, request: SearchRequest) -> SearchResponse:
        collection_metadata, embedding_provider = self._collection_context(
            request.collection_name
        )
        vector = self._embed_source(embedding_provider, request.source)
        return self._search_response(
            collection_metadata=collection_metadata,
            vector=vector,
            top_k=request.top_k,
            filters=request.filters,
            min_score=request.min_score,
        )

    def search_upload(
        self,
        *,
        collection_name: str,
        image_bytes: bytes,
        filename: str,
        top_k: Optional[int],
        min_score: Optional[float] = None,
    ) -> SearchResponse:
        collection_metadata, embedding_provider = self._collection_context(
            collection_name
        )

        validated_image = validate_and_load_image(
            image_bytes,
            max_size_mb=self.settings.max_image_size_mb,
            extension=filename,
            allowed_extensions=self.settings.allowed_image_extensions,
            max_pixels=self.settings.max_image_pixels,
        )
        try:
            vector = embedding_provider.embed_image(validated_image.image)
        finally:
            validated_image.image.close()
        return self._search_response(
            collection_metadata=collection_metadata,
            vector=vector,
            top_k=top_k,
            min_score=min_score,
        )

    def search_text(
        self,
        *,
        collection_name: str,
        query: str,
        top_k: Optional[int],
        min_score: Optional[float] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> SearchResponse:
        """Search stored images with words instead of a query image."""
        collection_metadata, embedding_provider = self._collection_context(
            collection_name
        )

        vector = embedding_provider.embed_text(query)
        return self._search_response(
            collection_metadata=collection_metadata,
            vector=vector,
            top_k=top_k,
            filters=filters,
            min_score=min_score,
        )

    def search_hybrid(
        self,
        *,
        collection_name: str,
        source: ImageSource,
        query: str,
        text_weight: float,
        top_k: Optional[int],
        min_score: Optional[float] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> SearchResponse:
        """Search with an image steered by a text description.

        Both encoders write into one vector space, so a weighted blend of the
        two query vectors expresses "this image, but ..." in a single search.
        """
        collection_metadata, embedding_provider = self._collection_context(
            collection_name
        )

        image_vector = self._embed_source(embedding_provider, source)
        text_vector = embedding_provider.embed_text(query)

        return self._search_response(
            collection_metadata=collection_metadata,
            vector=self._blend_vectors(image_vector, text_vector, text_weight),
            top_k=top_k,
            filters=filters,
            min_score=min_score,
        )

    def search_batch(
        self,
        *,
        collection_name: str,
        sources: list[ImageSource],
        mode: str,
        top_k: Optional[int],
        min_score: Optional[float] = None,
        filters: Optional[dict[str, object]] = None,
    ) -> BatchSearchResponse:
        resolved_top_k = self._resolve_top_k(top_k)

        collection_metadata, embedding_provider = self._collection_context(
            collection_name
        )
        vectors = [self._embed_source(embedding_provider, source) for source in sources]

        if mode == "average":
            average_vector = self._normalize_vector(self._average_vectors(vectors))
            results = self.vector_service.search(
                collection_name=collection_metadata.collection_name,
                vector=average_vector,
                top_k=resolved_top_k,
                filters=filters or {},
                min_score=min_score,
            )
            return BatchSearchResponse(
                collection_name=collection_metadata.collection_name,
                mode="average",
                top_k=resolved_top_k,
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
                    top_k=resolved_top_k,
                    filters=filters or {},
                    min_score=min_score,
                ),
            )
            for index, (source, vector) in enumerate(zip(sources, vectors))
        ]
        return BatchSearchResponse(
            collection_name=collection_metadata.collection_name,
            mode="separate",
            top_k=resolved_top_k,
            results=[],
            groups=groups,
        )

    def _collection_context(
        self,
        collection_name: str,
    ) -> tuple[CollectionMetadata, EmbeddingProvider]:
        collection_metadata = self.metadata_service.get(collection_name)
        return (
            collection_metadata,
            get_configured_provider(self.embedding_manager, collection_metadata),
        )

    def _search_response(
        self,
        *,
        collection_metadata: CollectionMetadata,
        vector: list[float],
        top_k: Optional[int],
        min_score: Optional[float] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> SearchResponse:
        resolved_top_k = self._resolve_top_k(top_k)
        results = self.vector_service.search(
            collection_name=collection_metadata.collection_name,
            vector=vector,
            top_k=resolved_top_k,
            filters=filters or {},
            min_score=min_score,
        )
        return SearchResponse(
            collection_name=collection_metadata.collection_name,
            top_k=resolved_top_k,
            results=results,
        )

    def _embed_source(
        self,
        embedding_provider: EmbeddingProvider,
        source: ImageSource,
    ) -> list[float]:
        image = self.image_loader.load_from_source(source)
        try:
            return embedding_provider.embed_image(image)
        finally:
            image.close()

    def _resolve_top_k(self, top_k: Optional[int]) -> int:
        """Fall back to DEFAULT_TOP_K and refuse anything above MAX_TOP_K."""
        resolved_top_k = top_k or self.settings.default_top_k

        if resolved_top_k > self.settings.max_top_k:
            raise BadRequestError(
                message="top_k exceeds the configured maximum",
                code="TOP_K_TOO_LARGE",
                details={
                    "top_k": resolved_top_k,
                    "max_top_k": self.settings.max_top_k,
                },
            )

        return resolved_top_k

    def _blend_vectors(
        self,
        image_vector: list[float],
        text_vector: list[float],
        text_weight: float,
    ) -> list[float]:
        image_weight = 1.0 - text_weight
        return self._normalize_vector(
            [
                image_weight * image_value + text_weight * text_value
                for image_value, text_value in zip(image_vector, text_vector)
            ]
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
                message="Query vector cannot be zero",
                code="ZERO_QUERY_VECTOR",
                details={},
            )
        return [value / norm for value in vector]
