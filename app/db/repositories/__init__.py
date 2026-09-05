"""Tenant-scoped data access.

Every repository except the tenant and API key ones requires a tenant id at
construction, so a query cannot be written without a scope.
"""

from app.db.repositories.products import ProductImageRepository, ProductRepository
from app.db.repositories.sources import SourceRepository
from app.db.repositories.tenants import ApiKeyRepository, TenantRepository
from app.db.repositories.vectors import VectorRepository

__all__ = [
    "ApiKeyRepository",
    "ProductImageRepository",
    "ProductRepository",
    "SourceRepository",
    "TenantRepository",
    "VectorRepository",
]
