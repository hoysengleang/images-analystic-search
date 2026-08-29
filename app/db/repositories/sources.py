from __future__ import annotations

import sqlite3
from typing import Optional

from app.db.repositories._mapping import from_iso, from_json, to_iso, to_json
from app.db.repositories._scoped import TenantScopedRepository
from app.models import Source, SourceStatus


class SourceRepository(TenantScopedRepository):
    def create(self, source: Source) -> Source:
        self._require_same_tenant(source.tenant_id)
        self.connection.execute(
            "INSERT INTO sources"
            " (id, tenant_id, type, name, config, status, sync_cursor,"
            "  last_synced_at, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source.id,
                self.tenant_id,
                source.type,
                source.name,
                to_json(source.config),
                source.status.value,
                source.sync_cursor,
                to_iso(source.last_synced_at),
                to_iso(source.created_at),
            ),
        )
        return source

    def get(self, source_id: str) -> Optional[Source]:
        row = self.connection.execute(
            "SELECT * FROM sources WHERE id = ? AND tenant_id = ?",
            (source_id, self.tenant_id),
        ).fetchone()
        return None if row is None else self._to_source(row)

    def list(self) -> list:
        rows = self.connection.execute(
            "SELECT * FROM sources WHERE tenant_id = ? ORDER BY created_at, id",
            (self.tenant_id,),
        ).fetchall()
        return [self._to_source(row) for row in rows]

    def update_sync_state(
        self,
        source_id: str,
        *,
        status: SourceStatus,
        sync_cursor: Optional[str] = None,
        last_synced_at=None,
    ) -> None:
        self.connection.execute(
            "UPDATE sources"
            " SET status = ?, sync_cursor = ?, last_synced_at = ?"
            " WHERE id = ? AND tenant_id = ?",
            (
                status.value,
                sync_cursor,
                to_iso(last_synced_at),
                source_id,
                self.tenant_id,
            ),
        )

    def delete(self, source_id: str) -> None:
        self.connection.execute(
            "DELETE FROM sources WHERE id = ? AND tenant_id = ?",
            (source_id, self.tenant_id),
        )

    def _require_same_tenant(self, tenant_id: str) -> None:
        if tenant_id != self.tenant_id:
            raise ValueError(
                f"Refusing to write tenant {tenant_id!r} data through a "
                f"repository scoped to {self.tenant_id!r}"
            )

    def _to_source(self, row: sqlite3.Row) -> Source:
        return Source(
            id=row["id"],
            tenant_id=row["tenant_id"],
            type=row["type"],
            name=row["name"],
            config=from_json(row["config"]),
            status=SourceStatus(row["status"]),
            sync_cursor=row["sync_cursor"],
            last_synced_at=from_iso(row["last_synced_at"]),
            created_at=from_iso(row["created_at"]),
        )
