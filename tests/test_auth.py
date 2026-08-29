"""API keys resolve to exactly one tenant, and nothing is stored in plaintext."""

import pytest

from app.auth import authenticate, generate_api_key, hash_api_key
from app.core.errors import UnauthorizedError
from app.db import transaction
from app.db.repositories import ApiKeyRepository, TenantRepository
from app.models import ApiKey, Tenant, TenantStatus


@pytest.fixture
def keys(db):
    """One tenant key, one admin key, one key for a suspended tenant."""
    plaintext = {
        "tenant": generate_api_key(),
        "admin": generate_api_key(),
        "suspended": generate_api_key(),
    }

    with transaction(db):
        tenants = TenantRepository(db)
        tenants.create(Tenant(id="t1", name="Shop One"))
        tenants.create(Tenant(id="t2", name="Shop Two", status=TenantStatus.SUSPENDED))

        api_keys = ApiKeyRepository(db)
        api_keys.create(
            ApiKey(
                id="k1",
                tenant_id="t1",
                name="Shop One key",
                key_hash=hash_api_key(plaintext["tenant"]),
            )
        )
        api_keys.create(
            ApiKey(
                id="k2",
                tenant_id=None,
                name="Admin key",
                key_hash=hash_api_key(plaintext["admin"]),
                is_admin=True,
            )
        )
        api_keys.create(
            ApiKey(
                id="k3",
                tenant_id="t2",
                name="Suspended key",
                key_hash=hash_api_key(plaintext["suspended"]),
            )
        )

    return db, plaintext


def resolve(db, presented):
    return authenticate(
        presented,
        api_key_repository=ApiKeyRepository(db),
        tenant_repository=TenantRepository(db),
    )


def test_generated_keys_are_unique_and_prefixed() -> None:
    generated = {generate_api_key() for _ in range(100)}

    assert len(generated) == 100
    assert all(key.startswith("ovs_") for key in generated)


def test_plaintext_keys_are_never_stored(keys) -> None:
    db, plaintext = keys
    stored = [row["key_hash"] for row in db.execute("SELECT key_hash FROM api_keys")]

    assert all(value not in stored for value in plaintext.values())
    assert all(len(value) == 64 for value in stored)


def test_a_tenant_key_resolves_to_its_tenant(keys) -> None:
    db, plaintext = keys

    principal = resolve(db, plaintext["tenant"])

    assert principal.tenant_id == "t1"
    assert principal.is_admin is False
    assert principal.require_tenant() == "t1"


def test_an_admin_key_owns_no_tenant_data(keys) -> None:
    db, plaintext = keys

    principal = resolve(db, plaintext["admin"])

    assert principal.is_admin is True
    assert principal.tenant_id is None
    with pytest.raises(UnauthorizedError) as exc_info:
        principal.require_tenant()
    assert exc_info.value.code == "TENANT_KEY_REQUIRED"


def test_an_admin_key_cannot_be_bound_to_a_tenant() -> None:
    with pytest.raises(ValueError, match="cannot belong to a tenant"):
        ApiKey(
            id="bad-admin",
            tenant_id="t1",
            name="Over-privileged admin",
            key_hash="hash",
            is_admin=True,
        )


def test_a_suspended_tenant_cannot_authenticate(keys) -> None:
    db, plaintext = keys

    with pytest.raises(UnauthorizedError) as exc_info:
        resolve(db, plaintext["suspended"])

    assert exc_info.value.code == "TENANT_SUSPENDED"


@pytest.mark.parametrize("presented", [None, "", "   ", "ovs_wrong", "not-a-key"])
def test_missing_or_unknown_keys_are_refused(keys, presented) -> None:
    db, _ = keys

    with pytest.raises(UnauthorizedError):
        resolve(db, presented)


def test_one_tenants_key_never_resolves_to_another_tenant(keys) -> None:
    db, plaintext = keys

    assert resolve(db, plaintext["tenant"]).tenant_id != "t2"


def test_successful_use_is_recorded(keys) -> None:
    db, plaintext = keys

    resolve(db, plaintext["tenant"])

    row = db.execute("SELECT last_used_at FROM api_keys WHERE id = 'k1'").fetchone()
    assert row["last_used_at"] is not None
