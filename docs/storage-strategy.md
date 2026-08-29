# Storage strategy

## What is stored

OpenVisionSearch is a search engine, not an image host. For each image it keeps:

```json
{
  "id": "shoe_001",
  "vector": [0.12, -0.44, 0.83, "…"],
  "payload": {
    "source_type": "url",
    "source_value": "https://cdn.example.com/shoe.jpg",
    "metadata": { "display_image_url": "https://cdn.example.com/shoe.jpg", "name": "Blue Shoe" },
    "embedding_provider": "openclip",
    "embedding_model": "ViT-B-32",
    "created_at": "2026-08-29T09:00:00+00:00"
  }
}
```

The decoded image is discarded once the vector exists. Your originals stay wherever they already live.

## Private buckets: sign, send, discard

For S3, R2, MinIO, Spaces, Wasabi, Firebase, or Supabase Storage:

```text
your backend generates a short-lived signed URL
  → POST it to OpenVisionSearch as a `url` source
    → the image is downloaded before the URL expires
      → the vector is stored, the bytes are dropped
```

**Do not store a signed URL as `display_image_url`.** It expires, and the stored value would then be dead. Store stable coordinates instead and re-sign at render time:

```json
{
  "id": "shoe_001",
  "source": { "type": "url", "value": "https://bucket.s3.amazonaws.com/products/shoe.jpg?X-Amz-Signature=…" },
  "metadata": {
    "storage_provider": "s3",
    "bucket": "team-a-products",
    "object_key": "products/shoe.jpg"
  }
}
```

When a search comes back, your backend reads `bucket` and `object_key` from the metadata and mints a fresh signed URL for the browser.

For public CDN images, `display_image_url` is simpler and works directly:

```json
{ "metadata": { "display_image_url": "https://cdn.example.com/shoe.jpg" } }
```

## Keeping the index in step with your database

Indexing is idempotent per id, so treat it as an upsert:

| Your event | Call |
| --- | --- |
| Product created | `POST /collections/{name}/index` with that product's id |
| Product image replaced | Same call with the same id — it overwrites |
| Product deleted | `DELETE /collections/{name}/images/{id}` |
| Bulk backfill | Page your table and send batches of images per request |

Use your own primary key as the image `id`. That way a search result maps straight back to a row without a lookup table.

## What persists where

| Data | Location | Survives a container restart |
| --- | --- | --- |
| Vectors and payloads | Qdrant (`qdrant_storage` volume) | Yes |
| Collection → model registry | `COLLECTION_METADATA_PATH` on the `./data` mount | Yes |
| Model weights | `model_cache` volume | Yes |
| Decoded images | Memory only, for the length of one request | No, by design |
