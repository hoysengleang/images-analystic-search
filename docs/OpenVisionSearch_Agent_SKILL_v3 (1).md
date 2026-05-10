---
name: open-vision-search-agent-skill
description: Build and maintain OpenVisionSearch, a free open-source FastAPI image similarity search API using OpenCLIP and Qdrant. Use this skill when creating project files, APIs, docs, tests, or architecture for the OpenVisionSearch project.
---

# OpenVisionSearch Agent Skill

## Mission

Build **OpenVisionSearch**, a free, open-source, self-hosted visual search API for developers.

Developers install the service, index their own images from different sources, and call HTTP endpoints to search visually similar images.

Default stack:

```text
FastAPI + OpenCLIP + Qdrant + Docker Compose
```

Default product rule:

```text
Store vectors and metadata by default, not original images.
```

## Product Promise

OpenVisionSearch lets developers index images from:

```text
URL
S3/R2 signed URL
file upload
local path inside container
mounted folder
base64
```

Then search by:

```text
URL
file upload
local path
base64
multiple images later
```

## Non-negotiable Architecture Rules

1. **Do not put model logic inside route files.**
2. **Do not put Qdrant logic inside route files.**
3. **Use provider/service layers.**
4. **OpenCLIP is the first provider, not the whole system.**
5. **Design for future model selection from the beginning.**
6. **One collection uses one embedding model.**
7. **Never mix vectors from different models inside one collection.**
8. **Do not store original images by default.**
9. **Support third-party storage through URL/signed URL first.**
10. **Keep API usable from every programming language through REST.**

## Core Data Flow

### Indexing

```text
Image source -> ImageLoader -> PIL image -> EmbeddingProvider -> vector -> Qdrant upsert with payload
```

### Searching

```text
Query image source -> ImageLoader -> EmbeddingProvider -> query vector -> Qdrant nearest-neighbor search -> JSON results
```

## Project Folder Structure

Create and maintain this structure:

```text
open-vision-search/
  app/
    main.py
    dependencies.py
    api/
      router.py
      routes/
        health.py
        collections.py
        index.py
        search.py
        images.py
        models.py
    core/
      config.py
      constants.py
      errors.py
      security.py
    schemas/
      collection.py
      image_source.py
      index.py
      search.py
      model.py
      common.py
    services/
      image_loader.py
      indexing_service.py
      search_service.py
      collection_service.py
      vector_service.py
      storage_service.py
    embedding/
      base.py
      registry.py
      manager.py
      providers/
        openclip_provider.py
        custom_provider.py
    providers/
      qdrant_provider.py
      storage/
        local_provider.py
        s3_provider.py
    utils/
      image_utils.py
      path_utils.py
      url_utils.py
      id_utils.py
  docs/
  examples/
  tests/
  data/
  Dockerfile
  docker-compose.yml
  requirements.txt
  .env.example
  README.md
  LICENSE
  CONTRIBUTING.md
```

## Required APIs

### Health

```http
GET /
GET /health
```

### Collections

```http
POST /collections
GET /collections
GET /collections/{collection_name}/stats
DELETE /collections/{collection_name}
```

### Models

```http
GET /models
```

### Indexing

```http
POST /collections/{collection_name}/index
POST /collections/{collection_name}/index/upload
POST /collections/{collection_name}/index/folder
```

### Search

```http
POST /collections/{collection_name}/search
POST /collections/{collection_name}/search/upload
POST /collections/{collection_name}/search/batch
```

### Image records

```http
GET /collections/{collection_name}/images/{image_id}
DELETE /collections/{collection_name}/images/{image_id}
```

## Image Source Schema

Use one flexible source object:

```json
{
  "source": {
    "type": "url",
    "value": "https://cdn.example.com/image.jpg"
  }
}
```

Allowed source types for v0.1:

```text
url
path
base64
```

Use multipart endpoints for upload.

Use folder endpoint for folder indexing.

## Storage Strategy

### Default

```text
Do not store original images.
Only store vector + metadata + source/display URL.
```

### URL / CDN / S3 signed URL

Flow:

```text
Download temporarily -> vectorize -> store vector + stable metadata -> discard image
```

Do not store expiring signed URL as permanent display URL.

Store stable data instead:

```json
{
  "display_image_url": "https://cdn.example.com/products/shoe.jpg",
  "storage_provider": "s3",
  "bucket": "team-a-products",
  "object_key": "products/shoe.jpg"
}
```

### Future direct S3 connector

Keep structure ready for:

```json
{
  "source": {
    "type": "s3",
    "bucket": "team-a-products",
    "key": "products/shoe.jpg"
  }
}
```

Do not implement direct S3 in v0.1 unless explicitly requested.

## Embedding Provider Design

Create base provider:

```python
class EmbeddingProvider:
    provider_name: str
    model_name: str
    vector_size: int

    def encode_image(self, image):
        raise NotImplementedError

    def encode_batch(self, images):
        return [self.encode_image(image) for image in images]
```

Providers:

```text
OpenCLIPProvider now
CustomModelProvider later
SigLIPProvider later
DINOv2Provider later
```

## Model Selection Rules

Every collection stores:

```text
embedding_provider
embedding_model
embedding_pretrained
vector_size
distance
```

When searching a collection:

```text
Read collection metadata -> select same provider/model -> encode query -> search Qdrant
```

Do not allow changing a collection model without re-indexing.

## Security Rules

### Local path

```text
Allow only paths under ALLOWED_IMAGE_ROOT.
Block ../ traversal.
Block paths outside the configured root.
Reject non-image files.
```

### URL

```text
Set timeout.
Limit redirects.
Reject huge files.
Validate content type.
Block private/internal IP ranges by default.
```

### Upload/base64

```text
Limit file size.
Allow jpg, jpeg, png, webp only.
Use Pillow to verify images.
Convert to RGB.
Reject broken images.
```

## Environment Variables

Use this `.env.example` pattern:

```env
APP_NAME=OpenVisionSearch
APP_HOST=0.0.0.0
APP_PORT=8000
APP_ENV=development

QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=

DEFAULT_EMBEDDING_PROVIDER=openclip
DEFAULT_MODEL_NAME=ViT-B-32
DEFAULT_MODEL_PRETRAINED=laion2b_s34b_b79k
DEFAULT_VECTOR_SIZE=512
DEFAULT_DISTANCE=cosine

ALLOWED_IMAGE_ROOT=/data/images
UPLOAD_DIR=/data/uploads
MAX_IMAGE_SIZE_MB=20
ALLOWED_IMAGE_TYPES=jpg,jpeg,png,webp

URL_DOWNLOAD_TIMEOUT_SECONDS=10
ALLOW_PRIVATE_URLS=false
MAX_REDIRECTS=3

STORE_ORIGINAL_IMAGES=false
STORAGE_DRIVER=none
LOCAL_STORAGE_PATH=/data/uploads

DEFAULT_TOP_K=10
MAX_TOP_K=100
```

## Implementation Order

Follow this order:

```text
1. FastAPI app + /health
2. Dockerfile + docker-compose with Qdrant
3. Pydantic settings config
4. Schemas for collection/model/image source
5. Qdrant provider
6. Collection service
7. Model registry with OpenCLIP only
8. Image loader for URL/path/base64/upload
9. Security checks for URL/path/upload
10. OpenCLIP provider
11. Indexing service and endpoint
12. Search service and endpoint
13. Folder indexing
14. Stats/delete endpoints
15. Tests
16. README and examples
```

## API Response Standards

Use consistent JSON:

Success:

```json
{
  "status": "success",
  "data": {}
}
```

Error:

```json
{
  "status": "error",
  "error": {
    "code": "INVALID_IMAGE_SOURCE",
    "message": "Image source is not supported."
  }
}
```

Search result:

```json
{
  "collection": "products",
  "results": [
    {
      "id": "product_001",
      "score": 0.93,
      "display_image_url": "https://cdn.example.com/products/shoe.jpg",
      "metadata": {}
    }
  ]
}
```

## Definition of Done for v0.1

```text
docker compose up works.
/docs opens successfully.
Can create a collection.
Can index URL, path, folder, base64, and upload images.
Can search URL, path, base64, and upload images.
Results return id, score, display_image_url, and metadata.
Qdrant stores vectors and payloads.
Original images are not stored by default.
README explains signed URL strategy for S3/R2/private storage.
Tests cover path safety and invalid images.
```

## Avoid

```text
Avoid SaaS-first design.
Avoid paid AI APIs in the core project.
Avoid cloud lock-in.
Avoid storing original images by default.
Avoid route-level business logic.
Avoid model-specific assumptions in API contracts.
```
