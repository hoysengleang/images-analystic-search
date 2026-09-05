# OpenVisionSearch Project Blueprint v3

> **Legacy reference.** This document describes the earlier image-similarity
> product. New work follows
> [`VISUAL_PRODUCT_SEARCH_SPEC.md`](VISUAL_PRODUCT_SEARCH_SPEC.md); consult the
> [roadmap](roadmap.md) for the transition status.

**Project type:** Free open-source developer tool  
**Main stack:** FastAPI + OpenCLIP + Qdrant  
**Main promise:** Developers can self-host a visual search API, index their own images from many sources, and search similar images through HTTP endpoints.

---

## 1. Product Mission

OpenVisionSearch is a free, open-source, self-hosted image similarity search service for developers.

It lets a developer team install the service inside their own environment, send images to be indexed, and later call an endpoint to find visually similar images.

The service is **not primarily an image storage system**. It is a **visual search engine**. By default it stores vectors and metadata, not original images.

### One-sentence README description

> OpenVisionSearch is a free, open-source, self-hosted visual search API. Developers can index images from URLs, uploads, local paths, folders, base64, S3 signed URLs, or other storage services, then search similar images through simple HTTP endpoints.

---

## 2. Target Users

| User Type | Why They Use It |
|---|---|
| Backend developers | Add image search to an existing app through REST API |
| E-commerce teams | Search similar products by image |
| Marketplace teams | Find related items, duplicates, and similar listings |
| CMS developers | Search media libraries visually |
| Internal tool builders | Add visual lookup without paying for cloud AI APIs |
| AI app builders | Use it as a reusable image embedding/search service |

---

## 3. Main Use Case Flow

```text
Team A has 100 product images.
They install OpenVisionSearch with Docker.
They index their images by URL, upload, path, folder, or base64.
OpenVisionSearch loads each image temporarily.
OpenCLIP converts each image into a vector embedding.
Qdrant stores image_id + vector + metadata.
Later, Team A user uploads a query image.
Team A backend calls OpenVisionSearch search endpoint.
OpenVisionSearch returns the most visually similar images.
```

---

## 4. Core Principle: Store Vectors, Not Images by Default

OpenVisionSearch should not force itself to become the customer's image storage platform.

Default behavior:

```text
Read image -> create vector -> store vector + metadata -> discard temporary image
```

Store in Qdrant:

```json
{
  "id": "product_001",
  "vector": [0.12, -0.44, 0.83],
  "payload": {
    "source_type": "url",
    "source_value": "https://cdn.team-a.com/products/shoe.jpg",
    "display_image_url": "https://cdn.team-a.com/products/shoe.jpg",
    "metadata": {
      "name": "Blue Shoe",
      "category": "shoes",
      "price": 29.99
    },
    "embedding": {
      "provider": "openclip",
      "model": "ViT-B-32",
      "vector_size": 512
    }
  }
}
```

Return in search result:

```json
{
  "results": [
    {
      "id": "product_001",
      "score": 0.93,
      "display_image_url": "https://cdn.team-a.com/products/shoe.jpg",
      "metadata": {
        "name": "Blue Shoe",
        "category": "shoes"
      }
    }
  ]
}
```

---

## 5. Image Source Support

The service should support many image input sources because different teams store images differently.

| Source Type | v0.1 Support | Purpose |
|---|---:|---|
| `url` | Yes | Public URL, CDN URL, S3 signed URL, R2 signed URL |
| `upload` | Yes | Multipart file upload |
| `path` | Yes | Local path inside service/container |
| `folder` | Yes | Batch index mounted folder |
| `base64` | Yes | Mobile/frontend systems that send base64 |
| `s3` direct connector | Later | Direct bucket/key access with credentials |
| `r2/minio/wasabi` direct connector | Later | S3-compatible storage support |

### Important rule for local path

Local path means a path visible to the OpenVisionSearch container/server, not the caller's laptop.

Example Docker mount:

```yaml
volumes:
  - ./product-images:/data/images
```

Allowed path example:

```text
/data/images/shoe.jpg
```

Blocked path examples:

```text
/etc/passwd
../../secret.txt
```

---

## 6. S3 and Third-party Storage Strategy

Many developer teams already store original images in S3, Cloudflare R2, MinIO, DigitalOcean Spaces, Wasabi, Firebase Storage, Supabase Storage, or a CDN.

### v0.1 approach: URL or signed URL first

For public images:

```json
{
  "id": "product_001",
  "source": {
    "type": "url",
    "value": "https://cdn.team-a.com/products/shoe.jpg"
  },
  "metadata": {
    "display_image_url": "https://cdn.team-a.com/products/shoe.jpg"
  }
}
```

For private S3/R2 images:

```text
Team A backend generates temporary signed URL.
Team A sends signed URL to OpenVisionSearch.
OpenVisionSearch downloads image before expiry.
OpenVisionSearch vectorizes the image.
OpenVisionSearch stores vector + stable metadata only.
```

Recommended metadata for private storage:

```json
{
  "metadata": {
    "display_image_url": "https://cdn.team-a.com/products/shoe.jpg",
    "storage_provider": "s3",
    "bucket": "team-a-products",
    "object_key": "products/shoe.jpg"
  }
}
```

Do not store short-lived signed URLs as permanent display URLs.

### Future direct storage connector

Future request example:

```json
{
  "source": {
    "type": "s3",
    "bucket": "team-a-products",
    "key": "products/shoe.jpg"
  }
}
```

Future config:

```env
STORAGE_CONNECTOR=s3
S3_ENDPOINT_URL=
S3_BUCKET=
S3_REGION=
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=
```

---

## 7. Search Algorithm Explanation

OpenVisionSearch uses image embeddings, not object detection, for the main search.

Algorithm:

```text
Image -> OpenCLIP image encoder -> vector embedding -> Qdrant nearest-neighbor search -> similar image results
```

Why images can be similar without being identical:

```text
The model compares visual and semantic meaning: shape, color, texture, object type, style, layout, and category.
A white sneaker and a red sneaker are not the same image, but their vectors may be close.
A sneaker and a backpack are far apart.
```

Two optional modes can exist later:

| Mode | Algorithm | Purpose |
|---|---|---|
| `similar` | OpenCLIP embeddings + Qdrant | Find visually/semantically similar images |
| `duplicate` | Perceptual hash | Find exact or near-duplicate files |

---

## 8. Model Strategy

### v0.1

Use OpenCLIP only.

### Future

Support model selection. Developers can choose the embedding model per collection.

Planned providers:

```text
openclip
custom
siglip
dinov2
mobileclip
```

Important rule:

```text
One collection must use one embedding model.
Do not mix vectors from different models in the same collection.
```

Because:

```text
OpenCLIP vector space != custom model vector space
```

Collection metadata must store:

```json
{
  "name": "products",
  "embedding_provider": "openclip",
  "embedding_model": "ViT-B-32",
  "embedding_pretrained": "laion2b_s34b_b79k",
  "vector_size": 512,
  "distance": "cosine"
}
```

---

## 9. Recommended Technical Stack

| Layer | Choice | Reason |
|---|---|---|
| API | FastAPI | Fast, clean, OpenAPI docs by default |
| Embedding | OpenCLIP | Free/open-source friendly image embeddings |
| Vector database | Qdrant | Fast, Docker-friendly, REST/gRPC, metadata payloads |
| Image processing | Pillow | Common Python image library |
| HTTP download | httpx | Async image download support |
| Config | Pydantic Settings | Clean environment-based config |
| Container | Docker Compose | Easy developer install |
| Tests | Pytest | Standard Python testing |

---

## 10. Project Folder Structure

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
    architecture.md
    api-reference.md
    image-sources.md
    storage-strategy.md
    model-selection.md
    deployment.md
    roadmap.md

  examples/
    curl/
    python/
    javascript/
    php/
    go/

  tests/
    test_health.py
    test_collections.py
    test_index_url.py
    test_index_path.py
    test_index_base64.py
    test_search.py
    test_security.py

  data/
    images/
    uploads/

  Dockerfile
  docker-compose.yml
  requirements.txt
  .env.example
  README.md
  LICENSE
  CONTRIBUTING.md
```

---

## 11. API Scope

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

### Images

```http
GET /collections/{collection_name}/images/{image_id}
DELETE /collections/{collection_name}/images/{image_id}
```

---

## 12. API Examples

### Create collection

```json
{
  "name": "products",
  "model": {
    "provider": "openclip",
    "name": "ViT-B-32",
    "pretrained": "laion2b_s34b_b79k"
  }
}
```

### Index mixed sources

```json
{
  "images": [
    {
      "id": "img_url_001",
      "source": {
        "type": "url",
        "value": "https://cdn.example.com/a.jpg"
      },
      "metadata": {
        "name": "Blue Shoe",
        "category": "shoes"
      }
    },
    {
      "id": "img_path_001",
      "source": {
        "type": "path",
        "value": "/data/images/b.jpg"
      },
      "metadata": {
        "name": "Local Product"
      }
    },
    {
      "id": "img_base64_001",
      "source": {
        "type": "base64",
        "value": "/9j/4AAQSkZJRgABAQ..."
      },
      "metadata": {
        "name": "Mobile Upload"
      }
    }
  ]
}
```

### Search by URL

```json
{
  "source": {
    "type": "url",
    "value": "https://cdn.example.com/query.jpg"
  },
  "top_k": 10,
  "min_score": 0.75
}
```

---

## 13. Environment Configuration

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

S3_ENDPOINT_URL=
S3_BUCKET=
S3_REGION=
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=

DEFAULT_TOP_K=10
MAX_TOP_K=100
```

---

## 14. Docker Compose

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./data/images:/data/images
      - ./data/uploads:/data/uploads
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

---

## 15. Security Requirements

### Path safety

```text
Only allow local paths inside ALLOWED_IMAGE_ROOT.
Block ../ traversal.
Block absolute paths outside configured root.
Reject non-image extensions.
```

### URL safety

```text
Set download timeout.
Limit redirects.
Reject huge files.
Validate content type.
Block private/internal IP ranges by default.
```

### Upload/base64 safety

```text
Limit file size.
Allow jpg, jpeg, png, webp only.
Verify image with Pillow.
Convert to RGB.
Reject broken images.
```

---

## 16. Build Roadmap

### v0.1 - Core MVP

```text
FastAPI server
Qdrant integration
OpenCLIP embedding provider
Create/list/delete collections
Index by URL, path, base64, upload, folder
Search by URL, path, base64, upload
Docker Compose
README quickstart
```

### v0.2 - Developer Experience

```text
Better examples for Python, JavaScript, PHP, Go
Metadata filtering
Batch indexing improvements
Multiple-image search
Cleaner error responses
```

### v0.3 - Production Features

```text
API key auth
Background indexing jobs
Indexing progress endpoint
Optional original-image storage drivers
Direct S3/R2/MinIO connector
GPU mode
Benchmarks
```

### v1.0 - Stable Open-source Release

```text
Stable API contract
Full documentation
Test coverage
Provider/plugin system
Model selection UI/API
Production deployment guide
```

---

## 17. First Implementation Order

```text
1. Create FastAPI project and /health endpoint
2. Add Dockerfile and docker-compose with Qdrant
3. Add config system with Pydantic Settings
4. Add Qdrant provider and collection service
5. Add model registry with OpenCLIP only
6. Add image source schemas
7. Add image loader for upload, URL, path, base64
8. Add path and URL security checks
9. Add OpenCLIP provider and embedding manager
10. Add index endpoint
11. Add search endpoint
12. Add folder indexing
13. Add image delete and stats endpoints
14. Add docs and examples
15. Add tests
```

---

## 18. Agent Implementation Rules

When an implementation agent builds this project, it must follow these rules:

```text
Do not make OpenCLIP logic directly inside route files.
Use service/provider structure.
Do not store original images by default.
Store model information per collection.
Never mix vectors from different models in one collection.
Validate and sanitize all image sources.
Keep all endpoints usable from any programming language.
Write examples in curl first, SDKs later.
Prefer simple MVP over over-engineering.
```

---

## 19. Definition of Done for v0.1

```text
A developer can run docker compose up.
A developer can open http://localhost:8000/docs.
A developer can create a collection.
A developer can index images by URL, upload, path, folder, and base64.
A developer can search by URL, upload, path, and base64.
The response returns image IDs, scores, display image URLs, and metadata.
The README explains S3/private storage signed URL usage.
The system stores vectors and metadata, not original images by default.
```
