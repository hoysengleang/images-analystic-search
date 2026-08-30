# 0006 — Native vectors load lazily per tenant and model

**Status:** Accepted · 2026-08-30 · Deviates from Section 9 startup wording

## Context

Section 9 says the native engine rebuilds its in-memory state at startup. A
single global rebuild would query across tenants, make startup time proportional
to every catalogue, and consume memory for tenants that receive no traffic.
The search contract already requires both tenant and model version on every
query, which gives the cache a safer partition boundary.

## Decision

SQLite remains the durable source of truth. The native engine loads a contiguous
`float32` matrix on the first search for one `(tenant_id, model_version)` pair.
Writes and deletes invalidate every cached model for only the affected tenant.
A restart begins with an empty cache and reconstructs each partition on demand.

Product metadata is loaded through a tenant-scoped repository for every search,
so stock, price, deletion, and other metadata changes do not require a vector
cache rebuild.

## Rationale

Lazy partitioning preserves the intended restart behavior without an unscoped
vector read. It reduces idle startup time and memory, while the first-query cost
is a predictable SQLite read plus one matrix allocation. This is the smallest
safe implementation for a single-node exact-search engine.

## Consequences

- The first query for a tenant/model pair is slower than later queries.
- No tenant's vectors are loaded as a side effect of another tenant's request.
- Multiple application processes build separate caches; SQLite remains shared
  durable state, but the first release should continue to use one API process.
- If cold-query latency becomes material, an authenticated tenant-scoped warmup
  operation can be added without changing the search contract.
