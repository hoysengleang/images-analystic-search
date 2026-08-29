# Architecture

> **In transition.** The project is moving from an image-similarity API to the
> multi-tenant product search platform described in
> `VISUAL_PRODUCT_SEARCH_SPEC.md`. Milestone 0 (the storage, tenancy, and
> engine foundation) is built and tested but **not yet wired into the running
> API** — the live endpoints are still the image-level `/collections` routes,
> which the `/v1` product API replaces in Milestone 1. See
> [the decision record](adr/0001-evolve-the-existing-repository.md).

## Foundation layers (Milestone 0)

| Package | Responsibility |
| --- | --- |
| `app/db` | SQLite connection setup, forward-only migrations, tenant-scoped repositories |
| `app/models` | Domain dataclasses: tenant, source, product, image, vector, job, analytics |
| `app/auth` | API key issuing and verification; resolves a request to one tenant |
| `app/search_engines` | The `SearchEngine` contract that the native engine and Qdrant both implement |
| `app/core/logging.py` | Structured JSON logs with credential redaction |

Tenant scope is bound when a repository is *constructed*, not passed per call,
so a query cannot be written without it. That is the mechanism behind the
specification's first priority; see
[ADR 0003](adr/0003-tenant-scope-in-repository-construction.md).

## Request flow (current image API)

```text
HTTP request
  └─ app/api/routes/*        thin FastAPI handlers, no business logic
      └─ app/services/*      the actual work
          ├─ ImageLoader     path / URL / base64 → validated RGB image
          ├─ EmbeddingManager → EmbeddingProvider (OpenCLIP) → vector
          └─ VectorService   Qdrant reads and writes
```

## Layers

| Package | Responsibility |
| --- | --- |
| `app/api` | Routing, request/response shapes, form parsing. No business logic. |
| `app/core` | Settings, constants, error types, API key check. |
| `app/schemas` | Pydantic request and response models. |
| `app/services` | Business logic: loading, indexing, searching, collections. |
| `app/embedding` | Model abstraction: base contract, registry, manager, providers. |
| `app/providers` | External systems, currently Qdrant. |
| `app/utils` | Pure helpers for images, paths, and URLs. |
| `app/dependencies.py` | The one place services are constructed and wired. |

## Dependency wiring

Service classes never build their own collaborators — every dependency is passed in. All wiring lives in `app/dependencies.py`, which means:

- routes depend on one well-known place,
- tests construct services directly with fakes,
- there is no import-time work and no circular imports.

## Indexing pipeline

```text
for each requested image:
    load and validate it       ← one failure only fails that image
group loaded images into batches of EMBED_BATCH_SIZE:
    one forward pass through the model
    one Qdrant upsert per batch
```

Per-image loading keeps a single broken file from failing the whole request. Batched embedding and storage is where the throughput comes from — the model forward pass dominates indexing time.

## Collection registry

Qdrant stores vectors; it does not track which embedding model produced them. Vectors from different models are not comparable, so the collection→model mapping is kept in a small JSON registry at `COLLECTION_METADATA_PATH` and consulted on every index and search call.

In Docker this file lives on the mounted `./data` volume so it survives restarts. If you run several API replicas, put it on shared storage.

## Adding an embedding model

1. Implement `EmbeddingProvider` in `app/embedding/providers/`, providing `embed_image`; override `embed_images` for a real batched path.
2. If the model also encodes text into the *same* vector space, set `supports_text = True` and implement `embed_text` (and `embed_texts` for batching). That is all the text and hybrid search endpoints need. Leave it alone otherwise and those endpoints return a clear `TEXT_SEARCH_NOT_SUPPORTED`.
3. Register the class in `app/embedding/registry.py`.
4. Create collections with that provider name and the model's true `vector_size`.

Nothing else changes: the manager caches provider instances per (provider, model, pretrained, vector size).

## Errors

Every error is returned in one shape:

```json
{ "error": { "code": "IMAGE_TOO_LARGE", "message": "…", "details": {} } }
```

Services raise typed exceptions from `app/core/errors.py`; handlers registered in `register_exception_handlers` turn them into that response.
