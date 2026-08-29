from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional, Protocol
from uuid import NAMESPACE_URL, uuid5

from app.core.errors import (
    BadRequestError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
from app.embedding.base import EmbeddingModelMetadata
from app.schemas.image_record import (
    ImageDeleteResponse,
    ImageListResponse,
    ImageRecordResponse,
)
from app.schemas.search import SearchResult

if TYPE_CHECKING:
    from qdrant_client.http import models as qdrant_models


#: Bounds accepted by a range filter, matching Qdrant's own operator names.
RANGE_OPERATORS = frozenset({"gt", "gte", "lt", "lte"})

#: Metadata keys Qdrant can address as a payload path. A key outside this set
#: makes Qdrant reject the whole query, which is a client mistake and must not
#: be reported as a backend outage.
FILTER_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*(\.[A-Za-z0-9_-]+)*$")


class VectorSource(Protocol):
    type: str
    value: str


@dataclass(frozen=True)
class VectorRecord:
    """One image ready to be written to Qdrant."""

    image_id: str
    vector: list
    source: VectorSource
    metadata: dict


class VectorService:
    """Every Qdrant point read/write goes through this service."""

    def __init__(self, qdrant_client: Any) -> None:
        self.qdrant_client = qdrant_client

    def upsert_image(
        self,
        *,
        collection_name: str,
        image_id: str,
        vector: list,
        source: VectorSource,
        metadata: Optional[dict] = None,
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
        records: Sequence,
        embedding: EmbeddingModelMetadata,
    ) -> list:
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
    ) -> dict:
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
        query_filter = self._build_filter(filters or {})

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
        count_filter = self._build_filter(filters or {})

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

    def _build_filter(
        self,
        filters: dict[str, Any],
    ) -> Optional[qdrant_models.Filter]:
        from qdrant_client.http import models as qdrant_models

        if not filters:
            return None

        conditions = [
            qdrant_models.FieldCondition(
                key=f"metadata.{self._validated_key(key)}",
                **self._condition_for(key, value),
            )
            for key, value in filters.items()
        ]

        return qdrant_models.Filter(must=conditions)

    def _validated_key(self, key: str) -> str:
        if not FILTER_KEY_PATTERN.match(key):
            raise self._unsupported_filter(
                key,
                None,
                "a filter field must be letters, digits, underscore, or hyphen, "
                "optionally dotted for nested fields",
            )
        return key

    def _condition_for(self, key: str, value: Any) -> dict[str, Any]:
        from qdrant_client.http import models as qdrant_models

        if isinstance(value, (bool, int, float, str)):
            return {"match": qdrant_models.MatchValue(value=value)}

        if isinstance(value, (list, tuple, set)):
            return {"match": qdrant_models.MatchAny(any=self._match_any(key, value))}

        if isinstance(value, dict):
            return {"range": qdrant_models.Range(**self._range_bounds(key, value))}

        raise self._unsupported_filter(
            key,
            value,
        )

    def _match_any(self, key: str, value: Any) -> list:
        values = list(value)
        all_strings = values and all(isinstance(item, str) for item in values)
        all_integers = values and all(
            isinstance(item, int) and not isinstance(item, bool) for item in values
        )

        if not (all_strings or all_integers):
            raise self._unsupported_filter(
                key,
                value,
                "a list filter must hold only strings or only integers",
            )

        return values

    def _range_bounds(self, key: str, value: dict[str, Any]) -> dict[str, Any]:
        unknown = set(value) - RANGE_OPERATORS
        if unknown or not value:
            raise self._unsupported_filter(
                key,
                value,
                f"range filters accept only {', '.join(sorted(RANGE_OPERATORS))}",
            )

        if not all(
            isinstance(bound, (int, float)) and not isinstance(bound, bool)
            for bound in value.values()
        ):
            raise self._unsupported_filter(key, value, "range bounds must be numbers")

        return dict(value)

    def _unsupported_filter(
        self,
        key: str,
        value: Any,
        reason: str,
    ) -> BadRequestError:
        return BadRequestError(
            message=f"Unsupported filter for '{key}': {reason}",
            code="UNSUPPORTED_FILTER",
            details={"field": key, "value": value},
        )

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
