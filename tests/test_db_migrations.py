"""The schema is the contract the rest of the storage layer relies on."""

import sqlite3

import pytest

from app.db import connect, migrate
from app.db.migrator import MigrationError, discover_migrations

EXPECTED_TABLES = {
    "api_keys",
    "feedback_events",
    "jobs",
    "product_images",
    "products",
    "schema_migrations",
    "search_events",
    "sources",
    "tenants",
    "vectors",
}


def test_migrations_create_the_full_schema(db) -> None:
    tables = {
        row["name"]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }

    assert tables >= EXPECTED_TABLES


def test_migrations_are_idempotent(db) -> None:
    assert migrate(db) == []


def test_migrations_record_what_they_applied(db) -> None:
    versions = [
        row["version"] for row in db.execute("SELECT version FROM schema_migrations")
    ]

    assert versions == ["0001"]


def test_foreign_keys_are_enforced(db) -> None:
    """Without this pragma SQLite silently ignores the schema's cascades."""
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO sources (id, tenant_id, type, name, created_at)"
            " VALUES ('s1', 'missing-tenant', 'local_manifest', 'Main', '2026-01-01')"
        )


def test_deleting_a_tenant_cascades_to_its_data(db) -> None:
    db.execute(
        "INSERT INTO tenants (id, name, status, created_at) VALUES ('t1','T','active','2026-01-01')"
    )
    db.execute(
        "INSERT INTO sources (id, tenant_id, type, name, created_at)"
        " VALUES ('s1','t1','local_manifest','Main','2026-01-01')"
    )

    db.execute("DELETE FROM tenants WHERE id = 't1'")

    assert db.execute("SELECT COUNT(*) AS n FROM sources").fetchone()["n"] == 0


def test_an_api_key_must_belong_to_a_tenant_unless_it_is_admin(db) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO api_keys (id, tenant_id, name, key_hash, is_admin, created_at)"
            " VALUES ('k1', NULL, 'orphan', 'hash', 0, '2026-01-01')"
        )


def test_an_admin_api_key_cannot_belong_to_a_tenant(db) -> None:
    db.execute(
        "INSERT INTO tenants (id, name, status, created_at)"
        " VALUES ('t1', 'Tenant', 'active', '2026-01-01')"
    )

    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO api_keys (id, tenant_id, name, key_hash, is_admin, created_at)"
            " VALUES ('k1', 't1', 'admin', 'hash', 1, '2026-01-01')"
        )


def test_a_product_is_unique_per_source_and_external_id(db) -> None:
    db.execute(
        "INSERT INTO tenants (id, name, status, created_at) VALUES ('t1','T','active','2026-01-01')"
    )
    db.execute(
        "INSERT INTO sources (id, tenant_id, type, name, created_at)"
        " VALUES ('s1','t1','local_manifest','Main','2026-01-01')"
    )
    insert = (
        "INSERT INTO products (id, tenant_id, source_id, external_id, attributes,"
        " content_hash, created_at, updated_at)"
        " VALUES (?, 't1', 's1', 'SHOE-1', '{}', 'h', '2026-01-01', '2026-01-01')"
    )
    db.execute(insert, ("p1",))

    with pytest.raises(sqlite3.IntegrityError):
        db.execute(insert, ("p2",))


def test_malformed_migration_filenames_are_rejected(tmp_path) -> None:
    (tmp_path / "not-a-migration.sql").write_text("SELECT 1;")

    with pytest.raises(MigrationError):
        discover_migrations(tmp_path)


def test_a_failed_migration_leaves_no_partial_schema(tmp_path) -> None:
    """Bookkeeping and DDL share one transaction, so neither lands alone."""
    (tmp_path / "0001_broken.sql").write_text(
        "CREATE TABLE good (id TEXT);\nCREATE TABLE bad (;"
    )
    connection = connect(tmp_path / "broken.db")

    with pytest.raises(MigrationError):
        migrate(connection, tmp_path)

    tables = {
        row["name"]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "good" not in tables
    assert (
        connection.execute("SELECT COUNT(*) AS n FROM schema_migrations").fetchone()["n"]
        == 0
    )
    connection.close()
