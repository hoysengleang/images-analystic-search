# 0003 — Tenant scope is bound at repository construction

**Status:** Accepted · 2026-08-29

## Context

Section 4 makes tenant isolation the highest priority, and Section 15 requires
tenant scope in storage rather than only in request handlers. The usual
approach — passing `tenant_id` to each method — relies on every author
remembering it, and a single omission leaks data across tenants.

## Decision

Every tenant-owned repository extends `TenantScopedRepository`, which takes
`tenant_id` in its constructor and rejects an empty one. Queries filter on
`self.tenant_id`. Writes additionally verify that the object's own `tenant_id`
matches, and raise rather than silently rewriting it.

`TenantRepository` and `ApiKeyRepository` are deliberately unscoped: they are
what establishes the scope.

## Rationale

A missing scope becomes impossible to express rather than easy to forget: there
is no constructor that omits the tenant, so a new query cannot be written
without one in hand.

## Consequences

- A repository instance is per-request and cheap; it holds only a connection
  and a string.
- Any operation genuinely spanning tenants must be written against the
  connection directly, which makes it conspicuous in review.
- Enforced by `tests/test_tenant_isolation.py`, which parametrises every scoped
  repository over reads, writes, listings, bulk lookups, and deletes.
