# API reference

Base URL: `http://localhost:8000` (plus `API_PREFIX` if set).

When `API_KEY` is configured, send `X-API-Key: <key>` on every request except `/health`.

Errors always come back in one shape:

```json
{ "error": { "code": "COLLECTION_NOT_FOUND", "message": "Collection not found: products", "details": {} } }
```

---

## `GET /health`

`search_engine` names the backend actually serving searches. `?include_qdrant=true` adds a Qdrant probe. The probe never fails the request — an unreachable Qdrant reports `"status": "unavailable"`.

```json
{ "app_name": "OpenVisionSearch", "status": "ok", "search_engine": "qdrant", "qdrant": { "status": "ok", "url": "http://qdrant:6333", "version": "1.12.0" } }
```

## `GET /models`

Lists the embedding models this server can serve, and which one new collections use by default. `supports_text` tells you whether the text and hybrid search endpoints will work on collections built with that model.

---

## `POST /collections`

```json
{
  "name": "products",
  "model": {
    "provider": "openclip",
    "name": "ViT-B-32",
    "pretrained": "laion2b_s34b_b79k",
    "vector_size": 512
  }
}
```

`model` also accepts `framing` (`pad` or `crop`) and `views` (1-3), which control how an image is fitted to the model input and how many views are averaged. Both are recorded with the collection and reused for every later index and search, because vectors framed differently are not comparable.

`model` is optional; omit it to use the server defaults. `vector_size` must match the model's real output size. **201** on success, **409** if the collection already exists.

## `GET /collections`

```json
{ "collections": [ { "name": "products", "model": { … }, "points_count": 0, "created_at": "2026-08-29T09:00:00+00:00" } ] }
```

## `GET /collections/{name}/stats`

```json
{ "name": "products", "points_count": 1240, "vector_size": 512, "distance": "cosine", "model": { … } }
```

## `DELETE /collections/{name}`

Deletes the Qdrant collection and its registry entry. Returns the deleted collection.

---

## `POST /collections/{name}/index`

```json
{
  "images": [
    { "id": "shoe_001", "source": { "type": "url", "value": "https://cdn.example.com/shoe.jpg" }, "metadata": { "name": "Blue Shoe" } },
    { "id": "shoe_002", "source": { "type": "path", "value": "/data/images/b.jpg" } },
    { "id": "shoe_003", "source": { "type": "base64", "value": "/9j/4AAQ…" } }
  ]
}
```

At most 500 images per request; send larger catalogues as several requests, or use folder indexing, which streams from disk with no such limit.

Response — partial success is normal, so check `errors`:

```json
{
  "collection_name": "products",
  "indexed_count": 2,
  "failed_count": 1,
  "ids": ["shoe_001", "shoe_002"],
  "errors": [ { "id": "shoe_003", "code": "INVALID_IMAGE", "message": "Invalid or broken image file", "details": {} } ]
}
```

Re-indexing an existing `id` overwrites that record.

## `POST /collections/{name}/index/upload`

`multipart/form-data` with `id` (required), `image` (required), `metadata` (optional JSON object).

## `POST /collections/{name}/index/folder`

```json
{ "folder_path": "/data/images/products", "recursive": true, "metadata": { "batch": "2026-08" } }
```

---

## `POST /collections/{name}/search`

```json
{
  "source": { "type": "url", "value": "https://cdn.example.com/query.jpg" },
  "top_k": 10,
  "min_score": 0.75,
  "filters": { "category": "shoes" }
}
```

`top_k` defaults to `DEFAULT_TOP_K` and is refused above `MAX_TOP_K`. `min_score` drops results below that similarity score.

### Searching part of the image

Every endpoint that takes a query image also accepts:

| Field | Type | Meaning |
| --- | --- | --- |
| `crop` | object | `{x, y, width, height}`, each a fraction of the image from 0 to 1. Searches only that region. |
| `detect` | boolean | Find the objects in the image and search each one. Requires `DETECTOR_PROVIDER`; defaults to false. |
| `detect_prompt` | list of strings | What to look for, for example `["a handbag"]`. Defaults to `DETECTOR_PROMPTS`. |

`crop` and `detect` are mutually exclusive. On `/search/upload` the crop arrives
as four separate form fields — `crop_x`, `crop_y`, `crop_width`, `crop_height` —
and all four are required together.

The response then carries:

| Field | Type | Meaning |
| --- | --- | --- |
| `query_regions` | list | The regions actually searched, each `{box, score, label}`. Empty when the whole image was used, including when detection found nothing. |
| `results[].matched_query_region` | integer or null | Index into `query_regions` naming the region that produced this hit. |

When several regions are searched, an image keeps its single best score across
them, and `min_score` applies to each region's score rather than to the merged
one. A detector that finds nothing falls back to searching the whole image.

### Filters

Filters match against the `metadata` you stored with each image. Three forms are accepted:

| Form | Example | Meaning |
| --- | --- | --- |
| Scalar | `{"category": "shoes"}` | Exact match on a string, number, or boolean |
| List | `{"category": ["shoes", "bags"]}` | Matches any value in the list |
| Range | `{"price": {"gte": 25, "lt": 100}}` | Numeric range; accepts `gt`, `gte`, `lt`, `lte` |

Field names must be letters, digits, underscore, or hyphen, optionally dotted for nested fields. Several filters combine with AND. A filter the server cannot apply is rejected with **400 `UNSUPPORTED_FILTER`** rather than ignored, so a search never silently returns unfiltered results.

```json
{
  "collection_name": "products",
  "top_k": 10,
  "results": [
    {
      "id": "shoe_001",
      "score": 0.93,
      "source_type": "url",
      "source_value": "https://cdn.example.com/shoe.jpg",
      "display_image_url": "https://cdn.example.com/shoe.jpg",
      "metadata": { "name": "Blue Shoe", "category": "shoes" }
    }
  ]
}
```

## `POST /collections/{name}/search/text`

Search stored images with words. No re-indexing is needed: OpenCLIP encodes text into the same vector space as the images, so a text vector can be compared directly against them.

```json
{ "query": "red running shoe with a white sole", "top_k": 10, "min_score": 0.2, "filters": { "category": "shoes" } }
```

The response is identical to `/search`.

**Scores are not on the same scale as image search.** A strong text match usually lands around 0.2–0.35, while a strong image match is 0.8 or above. That is the well-known gap between CLIP's text and image encoders, not a bug. Tune `min_score` separately for this endpoint, or leave it out and rely on `top_k`.

Returns **400 `TEXT_SEARCH_NOT_SUPPORTED`** if the collection's provider has no text encoder.

## `POST /collections/{name}/search/hybrid`

An image query steered by words — "this shoe, but blue".

```json
{
  "source": { "type": "url", "value": "https://cdn.example.com/shoe.jpg" },
  "query": "but in blue",
  "text_weight": 0.3,
  "top_k": 10
}
```

The image and text vectors are blended as `(1 - text_weight) x image + text_weight x text`, then re-normalized. `text_weight` of `0.0` is a plain image search and `1.0` is a plain text search; `0.2`–`0.4` keeps the image dominant while the words nudge the result. The response is identical to `/search`.

## `POST /collections/{name}/search/upload`

`multipart/form-data` with `image` (required), `top_k` and `min_score` (optional). This is the endpoint a phone "scan to search" flow calls.

## `POST /collections/{name}/search/batch`

```json
{
  "sources": [ { "type": "url", "value": "https://…/1.jpg" }, { "type": "url", "value": "https://…/2.jpg" } ],
  "mode": "average",
  "top_k": 10
}
```

At most 20 query images per request.

- `mode: "average"` — the query vectors are averaged and re-normalized into one query, returned in `results`. Use it when several photos show the same object.
- `mode: "separate"` — each query is searched on its own, returned in `groups` with the `source_index` it came from.

---

## `GET /collections/{name}/images`

`?limit=50&cursor=…`. Pages through stored records; pass the returned `next_cursor` to continue. `next_cursor` is `null` on the last page.

## `GET /collections/{name}/images/{id}`

```json
{
  "id": "shoe_001",
  "collection_name": "products",
  "source_type": "url",
  "source_value": "https://cdn.example.com/shoe.jpg",
  "metadata": { "name": "Blue Shoe" },
  "embedding_provider": "openclip",
  "embedding_model": "ViT-B-32",
  "created_at": "2026-08-29T09:00:00+00:00"
}
```

## `DELETE /collections/{name}/images/{id}`

**404** if the image is not in the collection.

---

## Error codes

Every failure returns the same envelope, so one handler in your client covers all of them:

```json
{ "error": { "code": "IMAGE_TOO_LARGE", "message": "...", "details": {} } }
```

Branch on `code`, not on `message` — messages are for humans and may be reworded.

### What each class of error means for you

| Status | Meaning | What to do |
| --- | --- | --- |
| 400 | The request or the image it points at is unusable | Fix the request; retrying unchanged will fail again |
| 401 | Missing or wrong `X-API-Key` | Check the key; do not retry |
| 404 | No such collection or image | Create it, or check the id |
| 409 | The name is taken | Pick another name, or reuse the existing collection |
| 413 | Body over `MAX_REQUEST_BODY_MB` | Send a smaller file |
| 422 | Body failed schema validation; `details.errors` lists the fields | Fix the payload |
| 503 | Qdrant or the model was unreachable | Safe to retry with backoff |

Only 503 is worth retrying automatically.

### Request and validation

| Code | Status | Cause |
| --- | --- | --- |
| `VALIDATION_ERROR` | 422 | Body did not match the schema |
| `METHOD_NOT_ALLOWED` | 405 | Right path, wrong HTTP method |
| `FORBIDDEN` | 403 | Refused by a proxy or middleware in front of the service |
| `HTTP_ERROR` | varies | An HTTP error with no more specific code |
| `BAD_REQUEST` | 400 | Generic bad request |
| `REQUEST_BODY_TOO_LARGE` | 413 | Body exceeds `MAX_REQUEST_BODY_MB` |
| `INVALID_UPLOAD_METADATA` | 400 | The `metadata` form field is not a JSON object |
| `TOP_K_TOO_LARGE` | 400 | `top_k` above `MAX_TOP_K` |
| `UNSUPPORTED_FILTER` | 400 | A filter field or value the server cannot apply |
| `ZERO_QUERY_VECTOR` | 400 | Blended query vectors cancelled out; change the sources or weight |

### Authentication

| Code | Status | Cause |
| --- | --- | --- |
| `UNAUTHORIZED` | 401 | Missing or unknown API key |
| `TENANT_KEY_REQUIRED` | 401 | An administrative key was used on a tenant endpoint |
| `TENANT_SUSPENDED` | 401 | The tenant this key belongs to is suspended |

### Collections and records

| Code | Status | Cause |
| --- | --- | --- |
| `COLLECTION_NOT_FOUND` | 404 | No such collection in the registry |
| `COLLECTION_ALREADY_EXISTS` | 409 | Name is taken |
| `IMAGE_NOT_FOUND` | 404 | No such image id in the collection |
| `NOT_FOUND` | 404 | Unknown route |
| `CONFLICT` | 409 | Generic conflict |
| `UNSUPPORTED_DISTANCE` | 400 | Not one of cosine, dot, euclid, manhattan |
| `UNKNOWN_EMBEDDING_PROVIDER` | 400 | No provider registered under that name |
| `TEXT_SEARCH_NOT_SUPPORTED` | 400 | The collection's model has no text encoder |
| `CROP_TOO_SMALL` | 400 | The crop rectangle covers too few pixels to search |
| `INCOMPLETE_CROP` | 400 | An upload sent some but not all of the four `crop_*` fields |
| `INVALID_CROP` | 400 | The `crop_*` fields are not a valid region of the image |
| `CROP_AND_DETECT_CONFLICT` | 400 | Send a crop or `detect`, not both |
| `DETECTION_NOT_AVAILABLE` | 400 | `detect` was asked for but `DETECTOR_PROVIDER` is `none` |
| `DETECTOR_PROMPTS_REQUIRED` | 400 | Detection needs at least one prompt saying what to find |
| `UNKNOWN_DETECTOR_PROVIDER` | 400 | `DETECTOR_PROVIDER` names a detector that is not registered |

### Images

| Code | Status | Cause |
| --- | --- | --- |
| `INVALID_IMAGE` | 400 | Bytes are not a decodable image |
| `IMAGE_TOO_LARGE` | 400 | Over `MAX_IMAGE_SIZE_MB` or `MAX_IMAGE_PIXELS` |
| `UNSUPPORTED_IMAGE_TYPE` | 400 | Format or content type not in `ALLOWED_IMAGE_TYPES` |
| `UNSUPPORTED_IMAGE_SOURCE` | 400 | `source.type` is not url, path, or base64 |
| `INVALID_BASE64_IMAGE` | 400 | Base64 payload included a `data:` prefix |
| `INVALID_IMAGE_PATH` | 400 | Path was empty |
| `IMAGE_PATH_NOT_ALLOWED` | 400 | Path resolves outside `ALLOWED_IMAGE_ROOT` |
| `INVALID_IMAGE_FOLDER` | 400 | Folder path is not a directory |

### Fetching a source URL

| Code | Status | Cause |
| --- | --- | --- |
| `INVALID_URL` | 400 | Missing or unparseable URL |
| `UNSUPPORTED_URL_SCHEME` | 400 | Not http or https |
| `PRIVATE_URL_NOT_ALLOWED` | 400 | Host is internal, or resolves to an internal address |
| `URL_HOST_RESOLUTION_FAILED` | 400 | Hostname does not resolve |
| `TOO_MANY_REDIRECTS` | 400 | More redirects than `MAX_REDIRECTS` |
| `INVALID_REDIRECT` | 400 | Redirect without a target |
| `IMAGE_URL_BAD_STATUS` | 400 | Source URL returned a non-2xx status |
| `INVALID_URL_TIMEOUT`, `INVALID_URL_REDIRECT_LIMIT` | 400 | Server misconfiguration; check the environment |
| `IMAGE_URL_DOWNLOAD_FAILED` | 503 | Network failure fetching the image |

### Backend failures — retry these

| Code | Status | Cause |
| --- | --- | --- |
| `QDRANT_*` | 503 | Qdrant unreachable or rejected the operation. The suffix names the operation: `VECTOR_SEARCH`, `VECTOR_UPSERT`, `VECTOR_DELETE`, `VECTOR_COUNT`, `IMAGE_RETRIEVE`, `IMAGE_LIST`, `CREATE_COLLECTION`, `DELETE_COLLECTION`, `COLLECTION_CHECK`, `COLLECTION_STATS` |
| `OPENCLIP_MODEL_LOAD_FAILED` | 503 | Weights could not be downloaded or loaded |
| `OPENCLIP_IMAGE_EMBEDDING_FAILED`, `OPENCLIP_TEXT_EMBEDDING_FAILED` | 503 | The model failed while encoding |
| `OPENCLIP_VECTOR_SIZE_MISMATCH` | 503 | `DEFAULT_VECTOR_SIZE` does not match the model's real output |
| `ONNX_MODEL_LOAD_FAILED` | 503 | The exported `.onnx` file is missing or could not be opened |
| `ONNX_TOKENIZER_LOAD_FAILED` | 503 | The tokenizer file is missing, unreadable, or is not a CLIP tokenizer |
| `ONNX_RUNTIME_NOT_INSTALLED`, `ONNX_TOKENIZER_NOT_INSTALLED` | 503 | Run `pip install -r requirements-onnx.txt` on the server |
| `ONNX_IMAGE_EMBEDDING_FAILED`, `ONNX_TEXT_EMBEDDING_FAILED` | 503 | The model failed while encoding |
| `ONNX_VECTOR_SIZE_MISMATCH` | 503 | `DEFAULT_VECTOR_SIZE` does not match the exported model's real output |
| `DETECTOR_MODEL_LOAD_FAILED`, `DETECTOR_TOKENIZER_LOAD_FAILED` | 503 | The detector files are missing or unreadable |
| `DETECTOR_INFERENCE_FAILED` | 503 | The detector failed while looking at the query image |
| `UNKNOWN_SEARCH_ENGINE` | 503 | `SEARCH_ENGINE` names a backend that is not registered |
| `SERVICE_UNAVAILABLE`, `INTERNAL_SERVER_ERROR` | 503 / 500 | Generic backend failure |
