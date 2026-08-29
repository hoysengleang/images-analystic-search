from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class EmbeddingStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Product:
    id: str
    tenant_id: str
    source_id: str
    external_id: str
    content_hash: str
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    in_stock: Optional[bool] = None
    attributes: dict = field(default_factory=dict)
    source_url: Optional[str] = None
    deleted_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None


@dataclass(frozen=True)
class ProductImage:
    id: str
    tenant_id: str
    product_id: str
    source_uri: str
    position: int = 0
    content_hash: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    media_type: Optional[str] = None
    embedding_status: EmbeddingStatus = EmbeddingStatus.PENDING
    model_version: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
