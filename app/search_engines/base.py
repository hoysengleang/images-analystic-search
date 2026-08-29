"""The contract every search backend implements.

The native engine and the Qdrant adapter are interchangeable behind this
interface. A backend owns vector persistence and nearest-neighbour retrieval
only; product grouping, ranking, and any other domain rule stays above it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class VectorSearchRequest:
    """A nearest-neighbour query.

    ``tenant_id`` and ``model_version`` are mandatory and have no defaults:
    an unscoped search, or one that mixes model versions, is never valid.
    """

    tenant_id: str
    model_version: str
    vector: list
    limit: int
    filters: dict = field(default_factory=dict)
    exclude_product_ids: tuple = ()

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.tenant_id.strip():
            raise ValueError("A vector search must name a tenant")
        if not self.model_version or not self.model_version.strip():
            raise ValueError("A vector search must name a model version")
        if not self.vector:
            raise ValueError("A vector search needs a query vector")
        if self.limit < 1:
            raise ValueError("A vector search needs a positive limit")


@dataclass(frozen=True)
class VectorHit:
    """One image-level match, before grouping into products."""

    vector_id: str
    product_id: str
    image_id: str
    score: float


@dataclass(frozen=True)
class EngineHealth:
    name: str
    healthy: bool
    vector_count: Optional[int] = None
    detail: Optional[str] = None


class SearchEngine(ABC):
    """Vector persistence and retrieval. No product-domain rules belong here."""

    name: str

    @abstractmethod
    def upsert(self, records: list) -> None:
        """Insert or replace vectors, keyed by their stable vector id."""

    @abstractmethod
    def delete(self, tenant_id: str, vector_ids: list) -> None:
        """Remove vectors belonging to one tenant."""

    @abstractmethod
    def delete_for_product(self, tenant_id: str, product_id: str) -> None:
        """Remove every vector belonging to one product."""

    @abstractmethod
    def search(self, request: VectorSearchRequest) -> list:
        """Return the nearest vectors, best score first."""

    @abstractmethod
    def health(self) -> EngineHealth:
        """Report whether the backend can serve traffic."""
