"""Tenant and API key storage.

These are the only repositories that are not tenant-scoped, because they are
what establishes the scope in the first place.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from app.db.repositories._mapping import from_iso, to_iso
from app.models import ApiKey, Tenant, TenantStatus


class TenantRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(self, tenant: Tenant) -> Tenant:
        self.connection.execute(
            "INSERT INTO tenants (id, name, status, created_at) VALUES (?, ?, ?, ?)",
            (
                tenant.id,
                tenant.name,
                tenant.status.value,
                to_iso(tenant.created_at),
            ),
        )
        return tenant

    def get(self, tenant_id: str) -> Optional[Tenant]:
        row = self.connection.execute(
            "SELECT * FROM tenants WHERE id = ?", (tenant_id,)
        ).fetchone()
        return None if row is None else self._to_tenant(row)

    def list(self) -> list:
        rows = self.connection.execute(
            "SELECT * FROM tenants ORDER BY created_at, id"
        ).fetchall()
        return [self._to_tenant(row) for row in rows]

    def _to_tenant(self, row: sqlite3.Row) -> Tenant:
        return Tenant(
            id=row["id"],
            name=row["name"],
            status=TenantStatus(row["status"]),
            created_at=from_iso(row["created_at"]),
        )


class ApiKeyRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(self, api_key: ApiKey) -> ApiKey:
        self.connection.execute(
            "INSERT INTO api_keys"
            " (id, tenant_id, name, key_hash, is_admin, created_at, last_used_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                api_key.id,
                api_key.tenant_id,
                api_key.name,
                api_key.key_hash,
                int(api_key.is_admin),
                to_iso(api_key.created_at),
                to_iso(api_key.last_used_at),
            ),
        )
        return api_key

    def get_by_hash(self, key_hash: str) -> Optional[ApiKey]:
        """Look up by hash: the plaintext key is never stored or compared."""
        row = self.connection.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
        ).fetchone()
        return None if row is None else self._to_api_key(row)

    def touch(self, api_key_id: str, used_at) -> None:
        self.connection.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
            (to_iso(used_at), api_key_id),
        )

    def list_for_tenant(self, tenant_id: str) -> list:
        rows = self.connection.execute(
            "SELECT * FROM api_keys WHERE tenant_id = ? ORDER BY created_at",
            (tenant_id,),
        ).fetchall()
        return [self._to_api_key(row) for row in rows]

    def _to_api_key(self, row: sqlite3.Row) -> ApiKey:
        return ApiKey(
            id=row["id"],
            tenant_id=row["tenant_id"],
            name=row["name"],
            key_hash=row["key_hash"],
            is_admin=bool(row["is_admin"]),
            created_at=from_iso(row["created_at"]),
            last_used_at=from_iso(row["last_used_at"]),
        )
