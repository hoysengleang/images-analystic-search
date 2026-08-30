from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional

from app.db.repositories._mapping import (
    from_bool,
    from_iso,
    from_json,
    to_bool,
    to_iso,
    to_json,
)
from app.db.repositories._scoped import TenantScopedRepository
from app.models import EmbeddingStatus, Product, ProductImage


class ProductRepository(TenantScopedRepository):
    def upsert(self, product: Product) -> Product:
        """Insert or update by (tenant, source, external id).

        Synchronisation replays the same products repeatedly, so writes have to
        be idempotent on the source's own identifier.
        """
        self._require_same_tenant(product.tenant_id)
        self.connection.execute(
            "INSERT INTO products"
            " (id, tenant_id, source_id, external_id, title, description, category,"
            "  brand, price, currency, in_stock, attributes, source_url,"
            "  content_hash, deleted_at, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (tenant_id, source_id, external_id) DO UPDATE SET"
            "  title = excluded.title,"
            "  description = excluded.description,"
            "  category = excluded.category,"
            "  brand = excluded.brand,"
            "  price = excluded.price,"
            "  currency = excluded.currency,"
            "  in_stock = excluded.in_stock,"
            "  attributes = excluded.attributes,"
            "  source_url = excluded.source_url,"
            "  content_hash = excluded.content_hash,"
            "  deleted_at = excluded.deleted_at,"
            "  updated_at = excluded.updated_at",
            (
                product.id,
                self.tenant_id,
                product.source_id,
                product.external_id,
                product.title,
                product.description,
                product.category,
                product.brand,
                product.price,
                product.currency,
                to_bool(product.in_stock),
                to_json(product.attributes),
                product.source_url,
                product.content_hash,
                to_iso(product.deleted_at),
                to_iso(product.created_at),
                to_iso(product.updated_at),
            ),
        )
        return self.get_by_external_id(product.source_id, product.external_id)

    def get(self, product_id: str) -> Optional[Product]:
        row = self.connection.execute(
            "SELECT * FROM products WHERE id = ? AND tenant_id = ?",
            (product_id, self.tenant_id),
        ).fetchone()
        return None if row is None else self._to_product(row)

    def get_by_external_id(
        self,
        source_id: str,
        external_id: str,
    ) -> Optional[Product]:
        row = self.connection.execute(
            "SELECT * FROM products"
            " WHERE tenant_id = ? AND source_id = ? AND external_id = ?",
            (self.tenant_id, source_id, external_id),
        ).fetchone()
        return None if row is None else self._to_product(row)

    def get_many(self, product_ids: list) -> dict:
        """Load several products at once, keyed by id, for result assembly."""
        if not product_ids:
            return {}

        placeholders = ",".join("?" for _ in product_ids)
        rows = self.connection.execute(
            f"SELECT * FROM products WHERE tenant_id = ? AND id IN ({placeholders})",
            (self.tenant_id, *product_ids),
        ).fetchall()
        return {row["id"]: self._to_product(row) for row in rows}

    def list(self, *, include_deleted: bool = False, limit: int = 100) -> list:
        query = "SELECT * FROM products WHERE tenant_id = ?"
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        query += " ORDER BY created_at, id LIMIT ?"

        rows = self.connection.execute(query, (self.tenant_id, limit)).fetchall()
        return [self._to_product(row) for row in rows]

    def content_hashes(self, source_id: str) -> dict:
        """External id to content hash, for skipping unchanged products."""
        rows = self.connection.execute(
            "SELECT external_id, content_hash FROM products"
            " WHERE tenant_id = ? AND source_id = ? AND deleted_at IS NULL",
            (self.tenant_id, source_id),
        ).fetchall()
        return {row["external_id"]: row["content_hash"] for row in rows}

    def mark_deleted(self, product_id: str, *, deleted_at=None) -> None:
        self.connection.execute(
            "UPDATE products SET deleted_at = ?, updated_at = ?"
            " WHERE id = ? AND tenant_id = ?",
            (
                to_iso(deleted_at or datetime.now(timezone.utc)),
                to_iso(datetime.now(timezone.utc)),
                product_id,
                self.tenant_id,
            ),
        )

    def delete(self, product_id: str) -> None:
        self.connection.execute(
            "DELETE FROM products WHERE id = ? AND tenant_id = ?",
            (product_id, self.tenant_id),
        )

    def _to_product(self, row: sqlite3.Row) -> Product:
        return Product(
            id=row["id"],
            tenant_id=row["tenant_id"],
            source_id=row["source_id"],
            external_id=row["external_id"],
            content_hash=row["content_hash"],
            title=row["title"],
            description=row["description"],
            category=row["category"],
            brand=row["brand"],
            price=row["price"],
            currency=row["currency"],
            in_stock=from_bool(row["in_stock"]),
            attributes=from_json(row["attributes"]),
            source_url=row["source_url"],
            deleted_at=from_iso(row["deleted_at"]),
            created_at=from_iso(row["created_at"]),
            updated_at=from_iso(row["updated_at"]),
        )


class ProductImageRepository(TenantScopedRepository):
    def upsert(self, image: ProductImage) -> ProductImage:
        self._require_same_tenant(image.tenant_id)
        self.connection.execute(
            "INSERT INTO product_images"
            " (id, tenant_id, product_id, source_uri, position, content_hash,"
            "  width, height, media_type, embedding_status, model_version, error,"
            "  created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (tenant_id, product_id, source_uri) DO UPDATE SET"
            "  position = excluded.position,"
            "  content_hash = excluded.content_hash,"
            "  width = excluded.width,"
            "  height = excluded.height,"
            "  media_type = excluded.media_type,"
            "  embedding_status = excluded.embedding_status,"
            "  model_version = excluded.model_version,"
            "  error = excluded.error,"
            "  updated_at = excluded.updated_at",
            (
                image.id,
                self.tenant_id,
                image.product_id,
                image.source_uri,
                image.position,
                image.content_hash,
                image.width,
                image.height,
                image.media_type,
                image.embedding_status.value,
                image.model_version,
                image.error,
                to_iso(image.created_at),
                to_iso(image.updated_at),
            ),
        )
        return self.get_by_uri(image.product_id, image.source_uri)

    def get(self, image_id: str) -> Optional[ProductImage]:
        row = self.connection.execute(
            "SELECT * FROM product_images WHERE id = ? AND tenant_id = ?",
            (image_id, self.tenant_id),
        ).fetchone()
        return None if row is None else self._to_image(row)

    def get_by_uri(self, product_id: str, source_uri: str) -> Optional[ProductImage]:
        row = self.connection.execute(
            "SELECT * FROM product_images"
            " WHERE tenant_id = ? AND product_id = ? AND source_uri = ?",
            (self.tenant_id, product_id, source_uri),
        ).fetchone()
        return None if row is None else self._to_image(row)

    def list_for_product(self, product_id: str) -> list:
        rows = self.connection.execute(
            "SELECT * FROM product_images"
            " WHERE tenant_id = ? AND product_id = ? ORDER BY position, id",
            (self.tenant_id, product_id),
        ).fetchall()
        return [self._to_image(row) for row in rows]

    def list_pending(self, limit: int = 100) -> list:
        rows = self.connection.execute(
            "SELECT * FROM product_images"
            " WHERE tenant_id = ? AND embedding_status = ?"
            " ORDER BY created_at, id LIMIT ?",
            (self.tenant_id, EmbeddingStatus.PENDING.value, limit),
        ).fetchall()
        return [self._to_image(row) for row in rows]

    def set_status(
        self,
        image_id: str,
        *,
        status: EmbeddingStatus,
        model_version: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self.connection.execute(
            "UPDATE product_images"
            " SET embedding_status = ?, model_version = ?, error = ?, updated_at = ?"
            " WHERE id = ? AND tenant_id = ?",
            (
                status.value,
                model_version,
                error,
                to_iso(datetime.now(timezone.utc)),
                image_id,
                self.tenant_id,
            ),
        )

    def delete_for_product(self, product_id: str) -> None:
        self.connection.execute(
            "DELETE FROM product_images WHERE tenant_id = ? AND product_id = ?",
            (self.tenant_id, product_id),
        )

    def _to_image(self, row: sqlite3.Row) -> ProductImage:
        return ProductImage(
            id=row["id"],
            tenant_id=row["tenant_id"],
            product_id=row["product_id"],
            source_uri=row["source_uri"],
            position=row["position"],
            content_hash=row["content_hash"],
            width=row["width"],
            height=row["height"],
            media_type=row["media_type"],
            embedding_status=EmbeddingStatus(row["embedding_status"]),
            model_version=row["model_version"],
            error=row["error"],
            created_at=from_iso(row["created_at"]),
            updated_at=from_iso(row["updated_at"]),
        )
