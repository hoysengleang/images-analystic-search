from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Protocol, TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from app.core.errors import ResourceNotFoundError, ServiceUnavailableError
from app.embedding.base import EmbeddingModelMetadata
from app.providers.qdrant_provider import get_qdrant_client
from app.schemas.image import ImageSource
from app.schemas.image_record import ImageDeleteResponse, ImageRecordResponse
from app.schemas.search import SearchResult

if TYPE_CHECKING:
    from qdrant_client.http import models as qdrant_models


class VectorSource(Protocol):
    type: str
    value: str


class VectorService:
    def __init__(self, qdrant_client: Optional[Any] = None) -> None:
        self.qdrant_client = qdrant_client or get_qdrant_client()

    def upsert_image(
        self,
        *,
        collection_name: str,
        image_id: str,
        vector: list[float],
        source: VectorSource,
        metadata: Optional[dict[str, Any]] = None,
        embedding: EmbeddingModelMetadata,
    ) -> str:
        from qdrant_client.http import models as qdrant_models

        point_id = self._point_id(collection_name=collection_name, image_id=image_id)
        payload = {
            "image_id": image_id,
            "source_type": source.type,
            "source_value": source.value,
            "metadata": metadata or {},
            "embedding_provider": embedding.provider,
            "embedding_model": embedding.model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
                wait=True,
            )
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not upsert image vector",
                code="QDRANT_VECTOR_UPSERT_FAILED",
                details={
                    "collection_name": collection_name,
                    "image_id": image_id,
                    "error": str(exc),
                },
            ) from exc

        return point_id

    def search(
        self,
        *,
        collection_name: str,
        vector: list[float],
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> list[SearchResult]:
        try:
            response = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=vector,
                query_filter=self._build_filter(filters or {}),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                score_threshold=min_score,
            )
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not search image vectors",
                code="QDRANT_VECTOR_SEARCH_FAILED",
                details={"collection_name": collection_name, "error": str(exc)},
            ) from exc

        points = getattr(response, "points", response)
        return [self._search_result_from_point(point) for point in points]

    def delete_image(self, *, collection_name: str, image_id: str) -> None:
        from qdrant_client.http import models as qdrant_models

        point_id = self._point_id(collection_name=collection_name, image_id=image_id)

        try:
            self.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=qdrant_models.PointIdsList(points=[point_id]),
                wait=True,
            )
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not delete image vector",
                code="QDRANT_VECTOR_DELETE_FAILED",
                details={
                    "collection_name": collection_name,
                    "image_id": image_id,
                    "error": str(exc),
                },
            ) from exc

    def get_image(
        self,
        *,
        collection_name: str,
        image_id: str,
    ) -> ImageRecordResponse:
        point = self._retrieve_point(collection_name=collection_name, image_id=image_id)
        return self._image_record_from_point(
            collection_name=collection_name,
            image_id=image_id,
            point=point,
        )

    def delete_image_record(
        self,
        *,
        collection_name: str,
        image_id: str,
    ) -> ImageDeleteResponse:
        self.get_image(collection_name=collection_name, image_id=image_id)
        self.delete_image(collection_name=collection_name, image_id=image_id)
        return ImageDeleteResponse(
            id=image_id,
            collection_name=collection_name,
            deleted=True,
        )

    def count_images(
        self,
        *,
        collection_name: str,
        filters: Optional[dict[str, Any]] = None,
    ) -> int:
        try:
            result = self.qdrant_client.count(
                collection_name=collection_name,
                count_filter=self._build_filter(filters or {}),
                exact=True,
            )
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not count image vectors",
                code="QDRANT_VECTOR_COUNT_FAILED",
                details={"collection_name": collection_name, "error": str(exc)},
            ) from exc

        return int(getattr(result, "count", result))

    def _build_filter(
        self,
        filters: dict[str, Any],
    ) -> Optional["qdrant_models.Filter"]:
        from qdrant_client.http import models as qdrant_models

        if not filters:
            return None

        conditions = [
            qdrant_models.FieldCondition(
                key=f"metadata.{key}",
                match=qdrant_models.MatchValue(value=value),
            )
            for key, value in filters.items()
            if isinstance(value, (bool, int, str))
        ]

        if not conditions:
            return None

        return qdrant_models.Filter(must=conditions)

    def _search_result_from_point(self, point: Any) -> SearchResult:
        payload = point.payload or {}
        metadata = payload.get("metadata") or {}

        return SearchResult(
            id=str(payload.get("image_id") or point.id),
            score=float(point.score),
            source_type=payload.get("source_type"),
            source_value=payload.get("source_value"),
            metadata=metadata,
            display_image_url=metadata.get("display_image_url"),
        )

    def _point_id(self, *, collection_name: str, image_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"{collection_name}:{image_id}"))

    def _retrieve_point(self, *, collection_name: str, image_id: str) -> Any:
        point_id = self._point_id(collection_name=collection_name, image_id=image_id)

        try:
            points = self.qdrant_client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not retrieve image vector payload",
                code="QDRANT_IMAGE_RETRIEVE_FAILED",
                details={
                    "collection_name": collection_name,
                    "image_id": image_id,
                    "error": str(exc),
                },
            ) from exc

        if not points:
            raise ResourceNotFoundError(
                message=f"Image not found: {image_id}",
                code="IMAGE_NOT_FOUND",
                details={
                    "collection_name": collection_name,
                    "image_id": image_id,
                },
            )

        return points[0]

    def _image_record_from_point(
        self,
        *,
        collection_name: str,
        image_id: str,
        point: Any,
    ) -> ImageRecordResponse:
        payload = point.payload or {}
        return ImageRecordResponse(
            id=str(payload.get("image_id") or image_id),
            collection_name=collection_name,
            source_type=payload.get("source_type"),
            source_value=payload.get("source_value"),
            metadata=payload.get("metadata") or {},
            embedding_provider=payload.get("embedding_provider"),
            embedding_model=payload.get("embedding_model"),
            created_at=payload.get("created_at"),
        )


def get_vector_service() -> VectorService:
    return VectorService()
