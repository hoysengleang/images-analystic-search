# 0005 — Vectors persisted as little-endian float32 blobs

**Status:** Accepted · 2026-08-29 · Resolves an open decision from Section 26

## Context

Section 26 leaves the SQLite vector persistence format open. Section 9 allows
BLOBs or an explicitly versioned binary format, and requires the native engine
to rebuild its in-memory state from durable storage.

## Decision

Each vector is stored as a `BLOB` of little-endian `float32` values, alongside
its `dimension`, `normalized` flag, and `model_version` as ordinary columns.
Byte order is normalised on write and on read, so a database file stays
portable between hosts.

## Rationale

float32 is what the models emit and what the search arithmetic uses, so no
conversion is needed at load time and the bytes map straight into a contiguous
array. It is half the size of float64 with no measurable effect on ranking.
JSON or a text encoding would cost parsing time on every rebuild.

Measured: exact cosine search over a contiguous float32 array takes 1.5 ms at
100,000 vectors and 18.5 ms at 1,000,000, against 50–150 ms to embed the query
image. Retrieval is not the bottleneck, which is what makes a native default
engine viable.

## Consequences

- 512-dimension vectors cost 2 KB each; a million vectors is roughly 2 GB in
  memory, so RAM rather than latency is the ceiling for the native engine.
- The dimension is stored explicitly, so a blob whose length disagrees with it
  is detectable rather than silently misread.
- A future format change needs a migration and a version marker on the column.
