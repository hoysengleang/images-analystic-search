# Deployment

## Docker Compose (recommended)

```bash
cp .env.example .env     # optional, defaults work
docker compose up --build -d
curl http://localhost:8000/health
```

The compose file gives you:

- the API on port 8000 and Qdrant reachable only on the internal Compose network,
- `./data/images` mounted read-only for merchant originals,
- an `app_state` volume for the writable collection registry,
- a `model_cache` volume so model weights are downloaded once, not on every rebuild,
- a `qdrant_storage` volume for the vectors,
- the API waiting for Qdrant's healthcheck before starting.

The API process runs as the unprivileged `openvisionsearch` user (UID 10001),
with a read-only root filesystem, all Linux capabilities dropped, and a small
temporary filesystem for uploaded request bodies. Qdrant is pinned to a specific
release so an ordinary rebuild cannot silently change the database version.

Qdrant's ports are intentionally not published on the host. If an operator
needs direct local access for maintenance, use a Compose override that publishes
port 6333 temporarily, then remove the override when finished.

### Moving an existing registry

Older Compose configurations stored the collection registry at
`./data/collections.json`. Before removing an old deployment, preserve that file.
After backing it up, initialize the new container and copy the registry into the
named volume before startup:

```bash
docker compose create api
docker compose cp ./data/collections.json api:/data/state/collections.json
docker compose up -d
```

Verify `GET /collections` before retiring the old copy. The `app_state` volume
then retains the existing collection-to-model mappings across container updates.

## First-request latency

The OpenCLIP weights (~600 MB for `ViT-B-32`) download on first use and take a few seconds to load. `WARMUP_MODEL_ON_STARTUP=true` (the compose default) moves that cost into container startup so the first real search is fast. Expect the very first `docker compose up` to spend a minute fetching weights.

## Sizing

| Load | Guidance |
| --- | --- |
| Indexing throughput | Raise `EMBED_BATCH_SIZE` (32–64 on a GPU). |
| Indexing memory | Images stream through the pipeline one batch at a time, so peak memory is `EMBED_BATCH_SIZE` x decoded image size, not catalogue size. A batch of 16 at ~2 MB each is ~32 MB, whether you index 100 images or 100,000. |
| CPU-only | Works fine for catalogues in the tens of thousands. Budget roughly 50–150 ms per image per core. |
| GPU | Install a CUDA build of torch in the image; the provider selects CUDA automatically when available. |
| Many collections | Each distinct model loads its own weights into memory. Prefer one model unless you truly need more. |

Vector storage is roughly `vector_size × 4 bytes` per image plus payload — about 2 KB per image for `ViT-B-32`.

## Running behind your own API

Keep OpenVisionSearch on a private network and call it from your backend. If you must expose it:

1. set `API_KEY` and send `X-API-Key`,
2. narrow `CORS_ALLOW_ORIGINS` to your real origins,
3. leave `ALLOW_PRIVATE_URLS=false` so callers cannot use the service to reach your internal network,
4. terminate TLS at your reverse proxy,
5. set `API_PREFIX` if you mount it under a path such as `/visual-search`.

## Multiple API replicas

Vectors are already shared, since every replica talks to the same Qdrant. The collection registry is a file, so replicas must share it — put `COLLECTION_METADATA_PATH` on a shared volume, or run a single API replica and scale later.

## Backups

- **Qdrant** — snapshot the `qdrant_storage` volume, or use Qdrant's own snapshot API.
- **Registry** — back up the `app_state` volume containing the JSON file at
  `COLLECTION_METADATA_PATH`. It is small and it is the only record of which
  model each collection uses.

Losing the registry does not lose your vectors, but nothing will know how to query them until it is restored.

## Upgrading the model

You cannot change a collection's model in place — existing vectors would be meaningless against new queries. Instead:

1. create a new collection with the new model,
2. re-index into it,
3. switch your application over,
4. delete the old collection.
