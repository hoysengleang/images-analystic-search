"""Minimal forward-only migration runner.

Numbered ``.sql`` files are applied in order inside one transaction each, and
recorded in ``schema_migrations``. A dedicated migration library would add a
dependency without buying anything at this size; see docs/adr/0002.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
MIGRATION_FILENAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")

CREATE_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    applied_at  TEXT NOT NULL
)
"""


class MigrationError(RuntimeError):
    """A migration file is malformed or failed to apply."""


def discover_migrations(migrations_dir: Path = MIGRATIONS_DIR) -> list:
    """Return (version, name, path) for every migration, in order."""
    migrations = []

    for path in sorted(migrations_dir.glob("*.sql")):
        match = MIGRATION_FILENAME.match(path.name)
        if match is None:
            raise MigrationError(
                f"Migration filename must look like 0001_description.sql: {path.name}"
            )
        migrations.append((match.group(1), path.stem, path))

    versions = [version for version, _, _ in migrations]
    duplicates = {version for version in versions if versions.count(version) > 1}
    if duplicates:
        raise MigrationError(f"Duplicate migration versions: {sorted(duplicates)}")

    return migrations


def applied_versions(connection: sqlite3.Connection) -> set:
    connection.execute(CREATE_MIGRATIONS_TABLE)
    rows = connection.execute("SELECT version FROM schema_migrations").fetchall()
    return {row["version"] for row in rows}


def migrate(
    connection: sqlite3.Connection,
    migrations_dir: Path = MIGRATIONS_DIR,
) -> list:
    """Apply every migration that has not run yet. Returns what was applied."""
    already_applied = applied_versions(connection)
    newly_applied = []

    for version, name, path in discover_migrations(migrations_dir):
        if version in already_applied:
            continue

        _apply(connection, version=version, name=name, path=path)
        newly_applied.append(name)

    return newly_applied


def _apply(
    connection: sqlite3.Connection,
    *,
    version: str,
    name: str,
    path: Path,
) -> None:
    """Run one migration and record it in the same transaction.

    The transaction lives inside the script because ``executescript`` commits
    anything pending before it runs, so an outer BEGIN would be discarded and
    a half-applied migration could be recorded as complete.
    """
    applied_at = datetime.now(timezone.utc).isoformat()
    script = "\n".join(
        [
            "BEGIN IMMEDIATE;",
            path.read_text(encoding="utf-8"),
            "INSERT INTO schema_migrations (version, name, applied_at)",
            f"VALUES ('{version}', '{name}', '{applied_at}');",
            "COMMIT;",
        ]
    )

    try:
        connection.executescript(script)
    except Exception as exc:
        # Must be execute, not executescript: executescript commits anything
        # pending before it runs, which would make the partial work permanent.
        with suppress(sqlite3.Error):
            connection.execute("ROLLBACK")
        raise MigrationError(f"Migration {name} failed: {exc}") from exc
