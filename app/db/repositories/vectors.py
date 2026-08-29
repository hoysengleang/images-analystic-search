"""Vector persistence for the native search engine.

Vectors are stored as raw little-endian float32 so they can be loaded straight
into a contiguous array at startup without per-row parsing.
"""

from __future__ import annotations

import array
import sqlite3
import sys
from typing import Optional

from app.db.repositories._mapping import from_iso, to_iso
from app.db.repositories._scoped import TenantScopedRepository
from app.models import VectorRecord

FLOAT32 = "f"


def pack_vector(values) -> bytes:
    packed = array.array(FLOAT32, values)
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def unpack_vector(blob: bytes) -> list:
    unpacked = array.array(FLOAT32)
    unpacked.frombytes(blob)
    if sys.byteorder != "little":
        unpacked.byteswap()
    return list(unpacked)


class VectorRepository(TenantScopedRepository):
    def upsert_many(self, records: list) -> int:
        for record in records:
            self._require_same_tenant(record.tenant_id)

        self.connection.executemany(
            "INSERT INTO vectors"
            " (id, tenant_id, product_id, image_id, model_version, dimension,"
            "  normalized, vector, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (tenant_id, id) DO UPDATE SET"
            "  product_id = excluded.product_id,"
            "  image_id = excluded.image_id,"
            "  model_version = excluded.model_version,"
            "  dimension = excluded.dimension,"
            "  normalized = excluded.normalized,"
            "  vector = excluded.vector,"
            "  created_at = excluded.created_at",
            [
                (
                    record.id,
                    self.tenant_id,
                    record.product_id,
                    record.image_id,
                    record.model_version,
                    record.dimension,
                    int(record.normalized),
                    pack_vector(record.vector),
                    to_iso(record.created_at),
                )
                for record in records
            ],
        )
        return len(records)

    def get(self, vector_id: str) -> Optional[VectorRecord]:
        row = self.connection.execute(
            "SELECT * FROM vectors WHERE id = ? AND tenant_id = ?",
            (vector_id, self.tenant_id),
        ).fetchone()
        return None if row is None else self._to_record(row)

    def list_for_model(self, model_version: str) -> list:
        rows = self.connection.execute(
            "SELECT * FROM vectors WHERE tenant_id = ? AND model_version = ? ORDER BY id",
            (self.tenant_id, model_version),
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def list_for_product(self, product_id: str) -> list:
        rows = self.connection.execute(
            "SELECT * FROM vectors WHERE tenant_id = ? AND product_id = ? ORDER BY id",
            (self.tenant_id, product_id),
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def delete(self, vector_ids: list) -> int:
        if not vector_ids:
            return 0

        placeholders = ",".join("?" for _ in vector_ids)
        cursor = self.connection.execute(
            f"DELETE FROM vectors WHERE tenant_id = ? AND id IN ({placeholders})",
            (self.tenant_id, *vector_ids),
        )
        return cursor.rowcount

    def delete_for_product(self, product_id: str) -> int:
        cursor = self.connection.execute(
            "DELETE FROM vectors WHERE tenant_id = ? AND product_id = ?",
            (self.tenant_id, product_id),
        )
        return cursor.rowcount

    def delete_for_image(self, image_id: str) -> int:
        cursor = self.connection.execute(
            "DELETE FROM vectors WHERE tenant_id = ? AND image_id = ?",
            (self.tenant_id, image_id),
        )
        return cursor.rowcount

    def count(self, model_version: Optional[str] = None) -> int:
        if model_version is None:
            row = self.connection.execute(
                "SELECT COUNT(*) AS total FROM vectors WHERE tenant_id = ?",
                (self.tenant_id,),
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT COUNT(*) AS total FROM vectors"
                " WHERE tenant_id = ? AND model_version = ?",
                (self.tenant_id, model_version),
            ).fetchone()
        return int(row["total"])

    def _require_same_tenant(self, tenant_id: str) -> None:
        if tenant_id != self.tenant_id:
            raise ValueError(
                f"Refusing to write tenant {tenant_id!r} data through a "
                f"repository scoped to {self.tenant_id!r}"
            )

    def _to_record(self, row: sqlite3.Row) -> VectorRecord:
        return VectorRecord(
            id=row["id"],
            tenant_id=row["tenant_id"],
            product_id=row["product_id"],
            image_id=row["image_id"],
            model_version=row["model_version"],
            dimension=row["dimension"],
            normalized=bool(row["normalized"]),
            vector=unpack_vector(row["vector"]),
            created_at=from_iso(row["created_at"]),
        )
