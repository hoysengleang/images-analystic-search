You are working on a free, open-source, self-hosted visual **product** search platform.

`VISUAL_PRODUCT_SEARCH_SPEC.md` (in the user's Codex outputs directory) is the source of truth. Developers connect a merchant catalogue, the platform indexes product images and metadata, and a shopper's photo returns visually similar **products** — not image rows.

The repository is mid-transition from its earlier image-similarity API. Milestone 0 (storage, tenancy, engine contract) is built; the live API is still the older `/collections` image routes until the `/v1` product API replaces them.

## Stack

- FastAPI for the API, Qdrant for vector search, OpenCLIP as the first embedding model, Pillow for decoding, httpx for downloads, Pydantic Settings for config, pytest for tests.

## Structure

```text
app/
  api/routes/     thin handlers: health, models, collections, index, search, images
  core/           config, constants, errors, security
  schemas/        request and response models
  services/       business logic
  embedding/      base contract, registry, manager, providers/
  providers/      Qdrant
  utils/          image, path, url helpers
  dependencies.py the single composition root
```

`docs/architecture.md` explains the layering; `CONTRIBUTING.md` has the conventions.

## Rules

- **Tenant isolation is priority #1.** Scope belongs in storage, not only in handlers: tenant-scoped repositories take the tenant at construction. Never write an unscoped query or vector search.
- **The native engine is the default.** Qdrant is optional and must never be required for `docker compose up`.
- Record the embedding model version with every vector, and never compare vectors from different versions.
- No OpenCLIP or Qdrant logic in route files. Routes validate, call a service, return.
- Services take every collaborator as an explicit keyword argument. All wiring lives in `app/dependencies.py`; nothing constructs its own dependencies.
- Do not store original images. Vectors and metadata only.
- Store the embedding model per collection, and never mix vectors from different models in one collection.
- Every image source goes through `ImageLoader` so it inherits the size, format, path, and URL safety checks.
- Raise the typed errors in `app/core/errors.py`; every response shares one error shape.
- Keep endpoints language-independent and easy to call from any backend.
- Prefer a simple MVP over over-engineering.
- Record decisions and deviations as ADRs in `docs/adr/`.
- Keep `python -m pytest` and `ruff check app tests examples` green.
