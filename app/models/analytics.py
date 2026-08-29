from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class FeedbackEventType(str, Enum):
    CLICK = "click"
    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"


@dataclass(frozen=True)
class SearchEvent:
    """Analytics for one search. Never holds the shopper's query image."""

    id: str
    tenant_id: str
    search_type: str
    filters: dict = field(default_factory=dict)
    result_count: int = 0
    latency_ms: Optional[int] = None
    engine: Optional[str] = None
    model_version: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class FeedbackEvent:
    id: str
    tenant_id: str
    search_id: str
    event: FeedbackEventType
    product_id: Optional[str] = None
    position: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
