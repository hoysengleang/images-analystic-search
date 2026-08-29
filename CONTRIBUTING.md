# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

Qdrant is not needed for the tests — they use fakes.

## Where to start reading

| If you want to... | Start at |
| --- | --- |
| Follow one request end to end | `app/api/routes/search.py` → `app/services/search_service.py` |
| See how services are built | `app/dependencies.py` — the single composition root |
| Understand image safety | `app/utils/image_utils.py` and `app/services/image_loader.py` |
| Swap in another model | `app/embedding/base.py`, then `providers/openclip_provider.py` |
| Understand tenant isolation | `app/db/repositories/_scoped.py` and [ADR 0003](docs/adr/0003-tenant-scope-in-repository-construction.md) |
| Know why something was done a certain way | `docs/adr/` |

`docs/architecture.md` maps the layers. The fastest way to see the whole API is
[`examples/curl/full-api-tour.sh`](examples/curl/full-api-tour.sh), which walks
every endpoint in the order you would use them.

## Conventions

- **Routes stay thin.** No business logic in `app/api/routes/`. Handlers validate input, call a service, and return its result.
- **Services take their dependencies explicitly.** No service constructs its own collaborators. Wiring belongs in `app/dependencies.py`.
- **Never mix models in a collection.** Everything about the embedding model is read from the collection registry.
- **Do not store original images.** Vectors and metadata only.
- **Validate every image source.** New sources go through `ImageLoader` so they inherit the size, format, path, and URL checks.
- **One error shape.** Raise the typed errors in `app/core/errors.py`; do not return bare `HTTPException`s with ad-hoc bodies.
- Type hints everywhere, keyword-only service arguments, and comments that explain *why* rather than restating the code.
- **Documentation is tested.** `tests/test_documentation_drift.py` asserts that every setting is documented and actually read, every endpoint appears in the README and the API reference, and every error code a caller can receive is listed. Adding a setting or an error code without documenting it fails the build — deliberately.

## Adding an endpoint

1. Add the request/response models to `app/schemas/`.
2. Add the logic to the relevant service in `app/services/`.
3. Add a thin handler to the matching module in `app/api/routes/`.
4. Cover the service logic and the route in `tests/`.
5. Update `docs/api-reference.md` and the endpoint table in the README.

## Adding an embedding provider

1. Implement `EmbeddingProvider` in `app/embedding/providers/`.
2. Override `embed_images` with a real batched implementation.
3. Register it in `app/embedding/registry.py`.
4. Add tests that do not download weights.

## Pull requests

Keep them focused, explain what breaks if the change is wrong, and make sure `python -m pytest` is green.
