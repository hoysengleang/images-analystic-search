from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Optional

from app.core.config import Settings
from app.core.errors import BadRequestError, ConflictError, ServiceUnavailableError
from app.embedding.manager import EmbeddingManager
from app.schemas.collection import (
    CollectionCreateRequest,
    CollectionModelConfig,
    CollectionResponse,
    CollectionStatsResponse,
)
from app.services.collection_metadata_service import (
    CollectionMetadata,
    CollectionMetadataService,
)

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


class CollectionService:
    """Creates and removes collections in Qdrant and in the local registry."""

    def __init__(
        self,
        *,
        settings: Settings,
        qdrant_client: QdrantClient,
        metadata_service: CollectionMetadataService,
        embedding_manager: EmbeddingManager,
    ) -> None:
        self.settings = settings
        self.qdrant_client = qdrant_client
        self.metadata_service = metadata_service
        self.embedding_manager = embedding_manager

    def create_collection(
        self,
        request: CollectionCreateRequest,
    ) -> CollectionResponse:
        model = self._resolve_model_config(request.model)
        collection_name = request.name.strip()

        self._ensure_collection_does_not_exist(collection_name)
        self._create_qdrant_collection(collection_name=collection_name, model=model)

        try:
            metadata = self.metadata_service.save(
                collection_name=collection_name,
                model=model,
            )
        except Exception:
            self._rollback_qdrant_collection(collection_name)
            raise

        return self._collection_response_from_metadata(
            metadata,
            points_count=0,
        )

    def list_collections(self) -> list[CollectionResponse]:
        return [
            self._collection_response_from_metadata(metadata)
            for metadata in self.metadata_service.list()
        ]

    def get_collection_stats(self, collection_name: str) -> CollectionStatsResponse:
        metadata = self.metadata_service.get(collection_name)
        points_count = self._get_qdrant_points_count(metadata.collection_name)

        return CollectionStatsResponse(
            name=metadata.collection_name,
            points_count=points_count,
            vector_size=metadata.vector_size,
            distance=metadata.distance,
            model=metadata.to_model_config(),
        )

    def delete_collection(self, collection_name: str) -> CollectionResponse:
        metadata = self.metadata_service.get(collection_name)
        points_count = self._get_qdrant_points_count(metadata.collection_name)

        try:
            self.qdrant_client.delete_collection(
                collection_name=metadata.collection_name,
            )
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not delete Qdrant collection",
                code="QDRANT_DELETE_COLLECTION_FAILED",
                details={
                    "collection_name": metadata.collection_name,
                    "error": str(exc),
                },
            ) from exc

        deleted = self.metadata_service.delete(metadata.collection_name)
        return self._collection_response_from_metadata(
            deleted,
            points_count=points_count,
        )

    def _resolve_model_config(
        self,
        model: Optional[CollectionModelConfig],
    ) -> CollectionModelConfig:
        if model is not None:
            return model

        metadata = self.embedding_manager.get_default_model_metadata()
        return CollectionModelConfig(
            provider=metadata.provider,
            name=metadata.model_name,
            pretrained=metadata.model_pretrained,
            vector_size=metadata.vector_size,
            distance=self.settings.default_distance,
            framing=self.settings.image_framing,
            views=self.settings.embed_views,
        )

    def _ensure_collection_does_not_exist(self, collection_name: str) -> None:
        try:
            exists = self.qdrant_client.collection_exists(
                collection_name=collection_name,
            )
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not check Qdrant collection",
                code="QDRANT_COLLECTION_CHECK_FAILED",
                details={"collection_name": collection_name, "error": str(exc)},
            ) from exc

        if exists:
            raise ConflictError(
                message=f"Collection already exists: {collection_name}",
                code="COLLECTION_ALREADY_EXISTS",
                details={"collection_name": collection_name},
            )

    def _create_qdrant_collection(
        self,
        *,
        collection_name: str,
        model: CollectionModelConfig,
    ) -> None:
        from qdrant_client.http import models as qdrant_models

        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=model.vector_size,
                    distance=self._qdrant_distance(model.distance),
                ),
                metadata={
                    "embedding_provider": model.provider,
                    "embedding_model": model.name,
                    "embedding_pretrained": model.pretrained,
                    "vector_size": model.vector_size,
                    "distance": model.distance,
                },
            )
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not create Qdrant collection",
                code="QDRANT_CREATE_COLLECTION_FAILED",
                details={"collection_name": collection_name, "error": str(exc)},
            ) from exc

    def _qdrant_distance(self, distance: str):
        from qdrant_client.http import models as qdrant_models

        distances = {
            "cosine": qdrant_models.Distance.COSINE,
            "dot": qdrant_models.Distance.DOT,
            "euclid": qdrant_models.Distance.EUCLID,
            "manhattan": qdrant_models.Distance.MANHATTAN,
        }

        try:
            return distances[distance]
        except KeyError as exc:
            raise BadRequestError(
                message=f"Unsupported distance metric: {distance}",
                code="UNSUPPORTED_DISTANCE",
                details={"distance": distance, "supported": sorted(distances)},
            ) from exc

    def _rollback_qdrant_collection(self, collection_name: str) -> None:
        # Best effort: the caller is already raising the real failure.
        with suppress(Exception):
            self.qdrant_client.delete_collection(collection_name=collection_name)

    def _get_qdrant_points_count(self, collection_name: str) -> int:
        try:
            info = self.qdrant_client.get_collection(collection_name=collection_name)
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not read Qdrant collection stats",
                code="QDRANT_COLLECTION_STATS_FAILED",
                details={"collection_name": collection_name, "error": str(exc)},
            ) from exc

        points_count = getattr(info, "points_count", None)
        if points_count is None:
            points_count = getattr(info, "vectors_count", 0)

        return int(points_count or 0)

    def _collection_response_from_metadata(
        self,
        metadata: CollectionMetadata,
        *,
        points_count: int = 0,
    ) -> CollectionResponse:
        return CollectionResponse(
            name=metadata.collection_name,
            model=metadata.to_model_config(),
            points_count=points_count,
            metadata={},
            created_at=metadata.created_at,
        )
