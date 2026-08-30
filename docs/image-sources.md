# Image sources

Every indexing and search endpoint accepts the same `source` object.

## URL

```json
{ "type": "url", "value": "https://cdn.example.com/shoe.jpg" }
```

Works for public CDN URLs and for signed S3/R2/MinIO/Spaces URLs. Safety rules applied to every download:

- scheme must be `http` or `https`,
- private, loopback, link-local, and reserved hosts are refused unless `ALLOW_PRIVATE_URLS=true`,
- the hostname is resolved and every address it points at is checked, so a public name whose DNS record aims at `127.0.0.1` or a cloud metadata endpoint is refused,
- redirects are followed by hand and **re-checked at every hop**, so a public URL cannot bounce the request onto an internal address,
- at most `MAX_REDIRECTS` redirects,
- `URL_DOWNLOAD_TIMEOUT_SECONDS` timeout,
- the response is streamed and aborted as soon as it passes `MAX_IMAGE_SIZE_MB`,
- HTML and other non-image content types are refused; `application/octet-stream` is allowed through because object stores commonly use it, and the bytes are then verified by the decoder.

## Upload

`multipart/form-data`, for phone cameras and browser file pickers.

```bash
curl -X POST http://localhost:8000/collections/products/index/upload \
  -F 'id=shoe_001' \
  -F 'image=@shoe.jpg' \
  -F 'metadata={"name":"Blue Shoe"}'
```

`metadata` is a JSON object sent as a form field.

## Local path

```json
{ "type": "path", "value": "/data/images/shoe.jpg" }
```

The path is resolved **inside the container**, not on the caller's machine, and must sit under `ALLOWED_IMAGE_ROOT`. Traversal (`../../etc/passwd`) and absolute paths outside the root are refused.

Mount your images with the volume already in `docker-compose.yml`:

```yaml
volumes:
  - ./data/images:/data/images:ro
```

The read-only mount is deliberate: indexing needs to read merchant originals,
but the service never needs to modify or delete them.

## Folder

Index everything under a mounted folder in one call:

```json
{ "folder_path": "/data/images/products", "recursive": true, "metadata": { "batch": "2026-08" } }
```

Image ids are derived from the path relative to the folder, with separators replaced by `__`, so `nested/b.png` becomes `nested__b`. Files whose extension is not in `ALLOWED_IMAGE_TYPES` are skipped silently; files that fail to decode are reported in `errors`.

## Base64

```json
{ "type": "base64", "value": "/9j/4AAQSkZJRgABAQ…" }
```

Send the raw base64 payload — a `data:image/jpeg;base64,` prefix is rejected so the encoding stays unambiguous.

## Metadata and display URLs

Metadata is stored next to the vector and returned with every search result, so your frontend can render results without a second lookup:

```json
{
  "id": "shoe_001",
  "source": { "type": "url", "value": "https://signed-url…" },
  "display_image_url": "https://cdn.example.com/shoe.jpg",
  "metadata": { "name": "Blue Shoe", "category": "shoes", "price": 29.99 }
}
```

Search results can be filtered on exact metadata values:

```json
{ "source": { … }, "filters": { "category": "shoes" } }
```

Filters accept a scalar (exact match), a list (match any), or a range object such as `{"price": {"gte": 25, "lt": 100}}`. A filter the server cannot apply is rejected rather than ignored — see [the API reference](api-reference.md) for the full syntax.
