from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Protocol
from uuid import NAMESPACE_URL, uuid5

from app.core.errors import ResourceNotFoundError, ServiceUnavailableError
from app.embedding.base import EmbeddingModelMetadata
from app.providers.qdrant_filters import build_qdrant_filter
from app.schemas.image_record import (
    ImageDeleteResponse,
    ImageListResponse,
    ImageRecordResponse,
)
from app.schemas.search import SearchResult


class VectorSource(Protocol):
    type: str
    value: str


@dataclass(frozen=True)
class VectorRecord:
    """One image ready to be written to Qdrant."""

    image_id: str
    vector: list[float]
    source: VectorSource
    metadata: dict[str, Any]


class VectorService:
    """Every Qdrant point read/write goes through this service."""

    def __init__(self, qdrant_client: Any) -> None:
        self.qdrant_client = qdrant_client

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
        point_ids = self.upsert_images(
            collection_name=collection_name,
            records=[
                VectorRecord(
                    image_id=image_id,
                    vector=vector,
                    source=source,
                    metadata=metadata or {},
                )
            ],
            embedding=embedding,
        )
        return point_ids[0]

    def upsert_images(
        self,
        *,
        collection_name: str,
        records: Sequence[VectorRecord],
        embedding: EmbeddingModelMetadata,
    ) -> list[str]:
        from qdrant_client.http import models as qdrant_models

        if not records:
            return []
        created_at = datetime.now(timezone.utc).isoformat()
        point_ids = [
            self._point_id(collection_name=collection_name, image_id=record.image_id)
            for record in records
        ]
        points = [
            qdrant_models.PointStruct(
                id=point_id,
                vector=record.vector,
                payload=self._build_payload(
                    record=record,
                    embedding=embedding,
                    created_at=created_at,
                ),
            )
            for point_id, record in zip(point_ids, records)
        ]
        try:
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not upsert image vectors",
                code="QDRANT_VECTOR_UPSERT_FAILED",
                details={
                    "collection_name": collection_name,
                    "image_ids": [record.image_id for record in records],
                    "error": str(exc),
                },
            ) from exc

        return point_ids

    def _build_payload(
        self,
        *,
        record: VectorRecord,
        embedding: EmbeddingModelMetadata,
        created_at: str,
    ) -> dict[str, Any]:
        return {
            "image_id": record.image_id,
            "source_type": record.source.type,
            "source_value": record.source.value,
            "metadata": record.metadata or {},
            "embedding_provider": embedding.provider,
            "embedding_model": embedding.model_name,
            # Revision and dimension are what make two vectors comparable;
            # without them a model upgrade silently corrupts a collection.
            "embedding_revision": embedding.model_pretrained,
            "vector_dimension": embedding.vector_size,
            "vector_normalized": True,
            "created_at": created_at,
        }

    def search(
        self,
        *,
        collection_name: str,
        vector: list[float],
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> list[SearchResult]:
        query_filter = build_qdrant_filter(filters or {})

        try:
            response = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=vector,
                query_filter=query_filter,
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

    def list_images(
        self,
        *,
        collection_name: str,
        limit: int,
        cursor: Optional[str] = None,
    ) -> ImageListResponse:
        try:
            points, next_offset = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=cursor,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not list image vectors",
                code="QDRANT_IMAGE_LIST_FAILED",
                details={"collection_name": collection_name, "error": str(exc)},
            ) from exc

        return ImageListResponse(
            collection_name=collection_name,
            images=[
                self._image_record_from_point(
                    collection_name=collection_name,
                    image_id=str(point.id),
                    point=point,
                )
                for point in points
            ],
            next_cursor=str(next_offset) if next_offset is not None else None,
        )

    def count_images(
        self,
        *,
        collection_name: str,
        filters: Optional[dict[str, Any]] = None,
    ) -> int:
        count_filter = build_qdrant_filter(filters or {})

        try:
            result = self.qdrant_client.count(
                collection_name=collection_name,
                count_filter=count_filter,
                exact=True,
            )
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not count image vectors",
                code="QDRANT_VECTOR_COUNT_FAILED",
                details={"collection_name": collection_name, "error": str(exc)},
            ) from exc

        return int(getattr(result, "count", result))

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
            embedding_revision=payload.get("embedding_revision"),
            vector_dimension=payload.get("vector_dimension"),
            created_at=payload.get("created_at"),
        )
