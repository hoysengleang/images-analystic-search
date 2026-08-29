# 0002 — Plain SQL migrations, no migration library

**Status:** Accepted · 2026-08-29

## Context

Section 9 requires migrations and foreign keys for the SQLite store.
Instruction 9 of Section 25 says not to add infrastructure until a measured
requirement justifies it.

## Decision

Numbered `.sql` files under `app/db/migrations/`, applied in order by a small
forward-only runner that records each version in a `schema_migrations` table.
No Alembic, no SQLAlchemy.

## Rationale

The runner is under a hundred lines and has no dependencies. Alembic's value is
autogeneration from an ORM model, and there is no ORM here — the repositories
use plain SQL so that tenant scoping is visible in every query.

## Consequences

- Migrations are forward-only; a mistake is corrected by a new migration.
- Each migration wraps its own transaction *inside* the script, because
  `sqlite3.executescript` commits anything pending before it runs. Rollback
  uses `execute`, not `executescript`, for the same reason. A test asserts that
  a failed migration leaves neither schema changes nor a bookkeeping row.
- If migrations ever need branching or data backfills across environments, this
  decision should be revisited.
