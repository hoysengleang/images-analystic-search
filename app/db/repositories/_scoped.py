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
