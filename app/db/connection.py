"""SQLite connection setup.

Every connection is configured the same way, because the pragmas below are
correctness settings rather than tuning: without ``foreign_keys`` SQLite
silently ignores the cascades the schema relies on.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

BUSY_TIMEOUT_MS = 5_000


def connect(database_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a configured connection, creating the database file if needed."""
    if database_path.parent and str(database_path) != ":memory:":
        database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(database_path),
        timeout=BUSY_TIMEOUT_MS / 1000,
        isolation_level=None,  # explicit transactions, no implicit BEGIN
        check_same_thread=False,
    )
    return configure(connection, read_only=read_only)


def configure(
    connection: sqlite3.Connection,
    *,
    read_only: bool = False,
) -> sqlite3.Connection:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    if not read_only:
        # WAL lets readers work while the sync worker writes.
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def transaction(connection: sqlite3.Connection):
    """Context manager for an explicit IMMEDIATE transaction."""
    return _Transaction(connection)


class _Transaction:
    """Commits on clean exit, rolls back on error.

    Nesting is handled by deferring to whichever block opened the transaction:
    an inner block is a no-op, so services can compose without coordinating.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self._owns_transaction = False

    def __enter__(self) -> sqlite3.Connection:
        self._owns_transaction = not self.connection.in_transaction
        if self._owns_transaction:
            # IMMEDIATE takes the write lock up front so two writers fail fast
            # instead of deadlocking part-way through.
            self.connection.execute("BEGIN IMMEDIATE")
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if not self._owns_transaction:
            return False

        if exc_type is None:
            self.connection.execute("COMMIT")
        else:
            self.connection.execute("ROLLBACK")
        return False


def close(connection: Optional[sqlite3.Connection]) -> None:
    if connection is not None:
        connection.close()
