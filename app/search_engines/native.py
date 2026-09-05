"""SQLite-backed exact vector search for the default single-node deployment.

The native engine deliberately favours correctness and a small dependency
surface over approximate-search complexity. Vectors remain durable in SQLite
and are cached per tenant and model version after their first query.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from threading import RLock
from typing import Any

import numpy as np

from app.db import transaction
from app.db.repositories import ProductRepository, VectorRepository
from app.models import Product, VectorRecord
from app.search_engines.base import (
    EngineHealth,
    SearchEngine,
    VectorHit,
    VectorSearchRequest,
)

RANGE_OPERATORS = frozenset({"gt", "gte", "lt", "lte"})
PRODUCT_FILTER_FIELDS = frozenset({"brand", "category", "in_stock", "price"})


@dataclass(frozen=True)
class _VectorIndex:
    vector_ids: tuple[str, ...]
    product_ids: tuple[str, ...]
    image_ids: tuple[str, ...]
    matrix: np.ndarray


class NativeSearchEngine(SearchEngine):
    """Persistent exact cosine search with mandatory tenant scope."""

    name = "native"

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self._cache: dict[tuple[str, str], _VectorIndex] = {}
        self._lock = RLock()

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return

        tenant_ids = {record.tenant_id.strip() for record in records}
        if "" in tenant_ids or len(tenant_ids) != 1:
            raise ValueError("A native-engine upsert must contain exactly one tenant")

        for record in records:
            if not record.model_version.strip():
                raise ValueError("A stored vector must name a model version")
            self._validated_vector(record.vector, expected_dimension=record.dimension)

        tenant_id = next(iter(tenant_ids))
        with self._lock, transaction(self.connection):
            repository = VectorRepository(self.connection, tenant_id=tenant_id)
            new_dimensions: dict[str, set[int]] = {}
            for record in records:
                new_dimensions.setdefault(record.model_version, set()).add(
                    record.dimension
                )
            for model_version, dimensions in new_dimensions.items():
                dimensions.update(repository.dimensions_for_model(model_version))
                if len(dimensions) != 1:
                    raise ValueError(
                        f"Model {model_version!r} cannot mix vector dimensions: "
                        f"{sorted(dimensions)}"
                    )

            repository.upsert_many(records)
            self._invalidate_tenant(tenant_id)

    def delete(self, tenant_id: str, vector_ids: list[str]) -> None:
        with self._lock, transaction(self.connection):
            VectorRepository(self.connection, tenant_id=tenant_id).delete(vector_ids)
            self._invalidate_tenant(tenant_id)

    def delete_for_product(self, tenant_id: str, product_id: str) -> None:
        with self._lock, transaction(self.connection):
            VectorRepository(self.connection, tenant_id=tenant_id).delete_for_product(
                product_id
            )
            self._invalidate_tenant(tenant_id)

    def search(self, request: VectorSearchRequest) -> list[VectorHit]:
        query, query_norm = self._validated_vector(request.vector)
        self._validate_filters(request.filters)

        with self._lock:
            index = self._vectors_for(request.tenant_id, request.model_version)
            product_ids = {
                product_id
                for product_id in index.product_ids
                if product_id not in request.exclude_product_ids
            }
            products = ProductRepository(
                self.connection, tenant_id=request.tenant_id
            ).get_many(sorted(product_ids))

        if not index.vector_ids:
            return []
        if index.matrix.shape[1] != len(query):
            raise ValueError(
                "Stored vector dimension does not match the query for model "
                f"{request.model_version!r}"
            )

        query_array = np.asarray(query, dtype=np.float32) / query_norm
        scores = index.matrix @ query_array
        hits = []
        for position, product_id in enumerate(index.product_ids):
            if product_id in request.exclude_product_ids:
                continue

            product = products.get(product_id)
            if (
                product is None
                or product.is_deleted
                or not self._matches_filters(product, request.filters)
            ):
                continue

            hits.append(
                VectorHit(
                    vector_id=index.vector_ids[position],
                    product_id=product_id,
                    image_id=index.image_ids[position],
                    score=float(scores[position]),
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.vector_id))
        return hits[: request.limit]

    def health(self) -> EngineHealth:
        try:
            self.connection.execute("SELECT 1").fetchone()
        except sqlite3.Error as exc:
            return EngineHealth(name=self.name, healthy=False, detail=str(exc))
        return EngineHealth(name=self.name, healthy=True)

    def _vectors_for(
        self,
        tenant_id: str,
        model_version: str,
    ) -> _VectorIndex:
        key = (tenant_id, model_version)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        records = VectorRepository(self.connection, tenant_id=tenant_id).list_for_model(
            model_version
        )
        if not records:
            loaded = _VectorIndex((), (), (), np.empty((0, 0), dtype=np.float32))
            self._cache[key] = loaded
            return loaded

        vectors = []
        dimensions = set()
        for record in records:
            vector, norm = self._validated_vector(
                record.vector, expected_dimension=record.dimension
            )
            vectors.append(np.asarray(vector, dtype=np.float32) / norm)
            dimensions.add(record.dimension)
        if len(dimensions) != 1:
            raise ValueError(
                f"Model {model_version!r} has inconsistent stored vector dimensions"
            )

        matrix = np.ascontiguousarray(np.stack(vectors), dtype=np.float32)
        matrix.flags.writeable = False
        loaded = _VectorIndex(
            vector_ids=tuple(record.id for record in records),
            product_ids=tuple(record.product_id for record in records),
            image_ids=tuple(record.image_id for record in records),
            matrix=matrix,
        )
        self._cache[key] = loaded
        return loaded

    def _invalidate_tenant(self, tenant_id: str) -> None:
        stale = [key for key in self._cache if key[0] == tenant_id]
        for key in stale:
            del self._cache[key]

    @staticmethod
    def _validated_vector(
        values: list,
        *,
        expected_dimension: int | None = None,
    ) -> tuple[tuple[float, ...], float]:
        if expected_dimension is not None and len(values) != expected_dimension:
            raise ValueError(
                f"Vector length {len(values)} does not match dimension "
                f"{expected_dimension}"
            )
        try:
            vector = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError("A vector may contain only numbers") from exc
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("A vector must contain finite numeric values")
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if norm == 0:
            raise ValueError("A zero vector cannot be searched or stored")
        return vector, norm

    @staticmethod
    def _validate_filters(filters: dict) -> None:
        for field, expected in filters.items():
            if field not in PRODUCT_FILTER_FIELDS and not field.startswith("attributes."):
                raise ValueError(f"Unsupported native-engine filter field: {field!r}")
            if field == "attributes." or ".." in field:
                raise ValueError(f"Invalid native-engine filter field: {field!r}")

            if isinstance(expected, (str, int, float, bool)):
                continue
            if isinstance(expected, (list, tuple)):
                if not expected or not all(
                    isinstance(value, type(expected[0])) for value in expected
                ):
                    raise ValueError(
                        f"List filter for {field!r} must be non-empty and homogeneous"
                    )
                continue
            if isinstance(expected, dict):
                if not expected or set(expected) - RANGE_OPERATORS:
                    raise ValueError(
                        f"Range filter for {field!r} accepts only "
                        f"{', '.join(sorted(RANGE_OPERATORS))}"
                    )
                if not all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in expected.values()
                ):
                    raise ValueError(f"Range filter for {field!r} requires numbers")
                continue
            raise ValueError(f"Unsupported native-engine filter value for {field!r}")

    @classmethod
    def _matches_filters(cls, product: Product, filters: dict) -> bool:
        return all(
            cls._matches(cls._product_value(product, field), expected)
            for field, expected in filters.items()
        )

    @staticmethod
    def _product_value(product: Product, field: str) -> Any:
        if field in PRODUCT_FILTER_FIELDS:
            return getattr(product, field)

        value: Any = product.attributes
        for component in field.removeprefix("attributes.").split("."):
            if not isinstance(value, dict) or component not in value:
                return None
            value = value[component]
        return value

    @staticmethod
    def _matches(actual: Any, expected: Any) -> bool:
        if isinstance(expected, dict):
            if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                return False
            checks = {
                "gt": lambda: actual > expected["gt"],
                "gte": lambda: actual >= expected["gte"],
                "lt": lambda: actual < expected["lt"],
                "lte": lambda: actual <= expected["lte"],
            }
            return all(checks[operator]() for operator in expected)
        if isinstance(expected, (list, tuple)):
            return actual in expected
        return actual == expected
