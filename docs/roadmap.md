# Roadmap

Aligned to the milestones in `VISUAL_PRODUCT_SEARCH_SPEC.md`.

## Milestone 0 — foundation (done)

- SQLite schema with foreign keys, and a forward-only migration runner
- Domain models for tenants, sources, products, images, vectors, jobs, analytics
- Tenant-scoped repositories: scope bound at construction, cross-tenant writes refused
- API keys stored as hashes, resolving to exactly one tenant
- The `SearchEngine` contract, with mandatory tenant and model version on every query
- Structured JSON logging with credential redaction
- CI on Python 3.9 and 3.12, and architecture decision records

## Milestone 1 — end-to-end native search (next)

- Local manifest connector reading JSON Lines plus a mounted image directory
- Native exact-search engine backed by SQLite (**engine complete; `/v1` wiring pending**)
- Catalogue synchronisation with content hashes, deletes, and job progress
- `/v1` product API on port 8080, replacing the image-level `/collections` routes
- Product grouping so results are products, not duplicate image rows
- Example catalogue that indexes with no external account

## Earlier image-layer work (carried forward)

## Shipped (v0.1 core)

- FastAPI service with Qdrant and OpenCLIP
- Collection create / list / stats / delete, with the embedding model pinned per collection
- Indexing by URL, upload, local path, folder, and base64
- Search by URL, upload, path, base64, and multi-image batch search
- Text-to-image search and hybrid image-plus-text search over the same index
- Aspect-preserving framing and optional multi-view embedding, pinned per collection
- Metadata filtering (scalar, list, and numeric range) and paginated record listing
- Path, URL, and image safety checks, including DNS resolution and per-hop redirect validation
- Optional API key auth and CORS
- Batched embedding and batched Qdrant writes
- Docker Compose with persistent volumes

## Next (v0.2 — developer experience)

- Query-side region search: crop rectangle and optional object detection (**shipped**)
- ONNX Runtime provider, so a server runs search without PyTorch (**shipped**)
- Scalar-rendered API reference at `/docs`, replacing Swagger UI (**shipped**)
- A demo page: drag an image in, see results, no client code needed
- Zero-shot labelling, scoring an image against caller-supplied labels — the text encoder is already loaded, so this is nearly free
- Client examples for Go and Ruby (curl, Python, JavaScript, and PHP are in `examples/`)
- Negation and nested metadata filters (scalar, list, and range filters are shipped)
- Payload indexes in Qdrant so filtered searches stay fast past a few hundred thousand images
- `PATCH` endpoint to update metadata without re-embedding
- Perceptual-hash duplicate detection alongside similarity search

## Later (v0.3 — production)

- Background indexing jobs with a progress endpoint
- Direct S3/R2/MinIO connectors, so callers pass a bucket and key instead of a signed URL
- Optional original-image storage drivers, off by default
- Structured request logging and Prometheus metrics
- Rate limiting
- GPU image and published benchmarks

## v1.0

- Frozen API contract
- More embedding providers (SigLIP, DINOv2, MobileCLIP)
- Collection registry backed by shared storage for multi-replica deployments
- Published client SDKs

## Known limits

- **Whole-image embedding on the index side.** A *query* photo can now be cropped, or detected and searched region by region, so a shopper's cluttered picture is handled. A cluttered *catalogue* photo is not: each indexed image is still one vector covering the whole frame. Indexing each crop needs product grouping first, or results fill with duplicates of the same photo — see [ADR 0007](adr/0007-query-side-region-search.md).
- **Synchronous indexing.** Large backfills are a series of HTTP calls from the client, not a job with a progress bar.
- **One API key per server.** There is no per-tenant key scoping yet.

## Deliberately not built yet

- **Image hosting.** The service stores vectors and metadata. Your originals stay where they are.
- **Mixed models per collection.** Vectors from different models are not comparable, so this stays forbidden.
- **A UI.** This is an API for developers to build on.
