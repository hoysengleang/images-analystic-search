"""API key issuing and verification.

Keys are stored only as SHA-256 hashes. A leaked database therefore does not
hand over working credentials, and the plaintext exists exactly once — in the
response to the call that created it.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from hmac import compare_digest
from typing import Optional

from app.core.errors import UnauthorizedError
from app.models import Tenant

API_KEY_PREFIX = "ovs_"
API_KEY_BYTES = 32


def generate_api_key() -> str:
    """Return a new plaintext key. Store only its hash."""
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(API_KEY_BYTES)}"


def hash_api_key(api_key: str) -> str:
    return sha256(api_key.strip().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """Who is making the request, and which tenant they may touch."""

    api_key_id: str
    tenant_id: Optional[str]
    is_admin: bool

    def require_tenant(self) -> str:
        """Return the tenant id, refusing administrative keys.

        Administrative keys manage tenants; they deliberately own no catalogue
        data, so they cannot stand in for a tenant on a search or index call.
        """
        if self.is_admin or self.tenant_id is None:
            raise UnauthorizedError(
                message="This endpoint requires a tenant-scoped API key",
                code="TENANT_KEY_REQUIRED",
            )
        return self.tenant_id


def authenticate(
    presented_key: Optional[str],
    *,
    api_key_repository,
    tenant_repository,
) -> AuthenticatedPrincipal:
    """Resolve a presented key to a principal, or refuse."""
    if not presented_key or not presented_key.strip():
        raise UnauthorizedError(message="An API key is required")

    stored_key = api_key_repository.get_by_hash(hash_api_key(presented_key))
    if stored_key is None:
        raise UnauthorizedError(message="Unknown API key")

    # The lookup already matched on the hash; this guards against a repository
    # that returns a near-miss row.
    if not compare_digest(stored_key.key_hash, hash_api_key(presented_key)):
        raise UnauthorizedError(message="Unknown API key")

    if stored_key.tenant_id is not None:
        tenant: Optional[Tenant] = tenant_repository.get(stored_key.tenant_id)
        if tenant is None:
            raise UnauthorizedError(message="Unknown API key")
        if not tenant.is_active:
            raise UnauthorizedError(
                message="This tenant is suspended",
                code="TENANT_SUSPENDED",
            )

    api_key_repository.touch(stored_key.id, datetime.now(timezone.utc))

    return AuthenticatedPrincipal(
        api_key_id=stored_key.id,
        tenant_id=stored_key.tenant_id,
        is_admin=stored_key.is_admin,
    )
