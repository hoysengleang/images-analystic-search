# 0001 — Evolve this repository rather than restart

**Status:** Accepted · 2026-08-29

## Context

The repository began as an image-similarity API: collections of images keyed by
arbitrary ids, Qdrant as a hard dependency, no tenants and no product concept.
The specification describes a multi-tenant *product* search platform with a
native default engine, a SQLite metadata store, and connector-driven
synchronisation.

The gap is structural, so the options were to restart or to evolve.

## Decision

Evolve this repository. Build the domain layer (tenants, products, sources,
jobs, engines) underneath the existing image and embedding code, and retire the
`/collections` API when the `/v1` product API replaces it.

## Rationale

The half already built is the half the specification's hardest sections depend
on: decode-don't-trust-headers image validation, EXIF handling, path
confinement, SSRF restriction with per-hop redirect checks, size and pixel
ceilings, batched embedding, and a model abstraction that records versions.
Sections 13 and 15 ask for precisely these, and they are already tested and
hardened. Rewriting them would reproduce the same code with fewer tests.

Nothing consumes the current API — the repository has one commit and no
release — so retiring `/collections` costs nothing.

## Consequences

- The directory layout moves toward Section 19 in stages rather than at once,
  so some modules sit in their old location until the milestone that moves them.
- `/collections` keeps working until the `/v1` API replaces it, then is removed
  in the same change rather than being maintained in parallel.
