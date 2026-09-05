"""Base class for repositories that may only ever see one tenant's data."""

from __future__ import annotations

import sqlite3


class TenantScopedRepository:
    """Binds a repository to one tenant at construction.

    Section 15 of the specification requires tenant scope in storage rather
    than only in request handlers. Taking the tenant here means no query in a
    subclass can be written without it — there is no unscoped constructor.
    """

    def __init__(self, connection: sqlite3.Connection, *, tenant_id: str) -> None:
        if not tenant_id or not tenant_id.strip():
            raise ValueError("A tenant-scoped repository requires a tenant id")

        self.connection = connection
        self.tenant_id = tenant_id.strip()

    def _require_same_tenant(self, tenant_id: str) -> None:
        """Refuse writes for data owned by another tenant."""
        if tenant_id != self.tenant_id:
            raise ValueError(
                f"Refusing to write tenant {tenant_id!r} data through a "
                f"repository scoped to {self.tenant_id!r}"
            )
