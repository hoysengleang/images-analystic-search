from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class SourceStatus(str, Enum):
    READY = "ready"
    SYNCING = "syncing"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass(frozen=True)
class Source:
    """A connected catalogue. ``config`` never holds secrets."""

    id: str
    tenant_id: str
    type: str
    name: str
    config: dict = field(default_factory=dict)
    status: SourceStatus = SourceStatus.READY
    sync_cursor: Optional[str] = None
    last_synced_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
