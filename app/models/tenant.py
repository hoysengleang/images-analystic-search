from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class TenantStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"


@dataclass(frozen=True)
class Tenant:
    id: str
    name: str
    status: TenantStatus = TenantStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_active(self) -> bool:
        return self.status is TenantStatus.ACTIVE


@dataclass(frozen=True)
class ApiKey:
    """A credential bound to exactly one tenant, or an administrative key.

    Only the hash is ever stored; the plaintext key exists once, at creation.
    """

    id: str
    name: str
    key_hash: str
    tenant_id: Optional[str] = None
    is_admin: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.is_admin and self.tenant_id is not None:
            raise ValueError("An administrative API key cannot belong to a tenant")
        if not self.is_admin and self.tenant_id is None:
            raise ValueError("A non-administrative API key must belong to a tenant")
