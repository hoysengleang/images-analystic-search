from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class JobType(str, Enum):
    SYNC_SOURCE = "sync_source"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class Job:
    id: str
    tenant_id: str
    type: JobType
    status: JobStatus = JobStatus.QUEUED
    source_id: Optional[str] = None
    progress: dict = field(default_factory=dict)
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
