# Open Visual Product Search

> Working product name. The name may change without changing this specification.

## 1. Document purpose

This document is the source of truth for designing and implementing an open-source, self-hosted visual product search platform.

It is written for maintainers, contributors, and coding agents. When implementation choices conflict with this document, follow the priorities in Section 4 and record any necessary deviation in an architecture decision record.

## 2. Product summary

Open Visual Product Search lets a developer connect a merchant's product catalogue, index its product images and metadata, and add image-based product discovery to an application.

A shopper uploads a photo or selects an existing product image. The platform returns visually similar products from that merchant's catalogue. Product images and metadata may originate from local folders, CSV or JSON files, databases, image URLs, S3-compatible storage, or cloud connectors.

The platform is not a general web image-search engine. It searches sources explicitly connected by the developer or merchant.

### Product statement

> Connect any product catalogue and add self-hosted visual product search through one API.

### Primary example

1. A merchant connects a product catalogue containing shoes.
2. The platform synchronizes product records and images.
3. A shopper uploads a photograph of a shoe.
4. The platform converts the photograph into a visual embedding.
5. It finds visually similar catalogue images, groups them by product, applies product filters, and reranks the candidates.
6. It returns products with their original merchant URLs.

## 3. Product goals

The platform must:

- Run locally through Docker Compose without a paid account or hosted dependency.
- Provide a native search engine enabled by default.
- Allow optional search backends, including Qdrant, without making them mandatory.
- Support image-to-product search.
- Support related-product search using an already indexed product.
- Keep product metadata linked to its original source.
- Isolate every merchant or tenant.
- Provide a stable REST API and a JavaScript client.
- Make connectors, embedding providers, and search engines replaceable.
- Record privacy-conscious search analytics and relevance feedback.

## 4. Priority order

When trade-offs are necessary, use this order:

1. Correct tenant isolation and security
2. Correct and explainable search results
3. Simple local developer experience
4. Reliable synchronization and deletion
5. Stable interfaces
6. Performance
7. Additional connectors and UI features

Do not trade tenant isolation or data correctness for performance.

## 5. Non-goals for the first release

The first release will not:

- Crawl the public internet.
- Search images that a developer has not connected.
- Train a foundation image model from scratch.
- Provide advertising, checkout, payment, or marketplace functionality.
- Build a distributed native vector database.
- Provide personalized recommendations based on shopper identity.
- Guarantee exact SKU recognition from every photograph.
- Require Qdrant, OpenSearch, Elasticsearch, or a managed vector service.

## 6. Users

### Application developer

The developer deploys the platform, connects a catalogue, calls the API, and embeds visual search into a website or mobile application.

### Merchant or catalogue administrator

The administrator monitors source synchronization, indexing errors, catalogue coverage, and search quality.

### Shopper

The shopper uploads an image, optionally selects or crops an object, filters results, and opens a matching product page.

## 7. Core user journeys

### 7.1 Connect and index a catalogue

1. Create a tenant.
2. Register a source.
3. Validate source configuration and access.
4. Read product records incrementally.
5. Validate and normalize each record.
6. Fetch and normalize product images.
7. Create embeddings for every usable image.
8. Store searchable vectors and product metadata.
9. Save a synchronization checkpoint.
10. Report indexed, skipped, failed, changed, and deleted records.

### 7.2 Search using an uploaded image

1. Authenticate the request and resolve the tenant.
2. Validate file type, size, and decoded image dimensions.
3. Normalize orientation and color mode.
4. Optionally apply a shopper-provided crop.
5. Generate a query embedding with the same model version used by the index.
6. Retrieve candidate images from the configured engine.
7. Apply tenant and product filters.
8. Group image matches into products.
9. Rerank products.
10. Return results, scores, source links, and a search identifier.

### 7.3 Find related products

1. Receive an indexed product ID.
2. Load the product's image embeddings.
3. Search using one or more representative vectors.
4. Exclude the original product.
5. Group, filter, rerank, and return related products.

### 7.4 Synchronize changes

- New products are indexed.
- Changed products or images are re-indexed.
- Unchanged content is skipped using stable hashes.
- Deleted source products are removed or tombstoned promptly.
- A failed run must not corrupt the previous searchable index.

## 8. System architecture

```text
Product sources
├── Local folder + manifest
├── CSV or JSON catalogue
├── Image URLs
├── S3-compatible object storage
├── Database
└── Future cloud connectors
          │
          ▼
Connector and synchronization layer
          │
          ▼
Product and image normalization
          │
          ▼
Local embedding provider
          │
          ▼
Search engine interface
├── Native exact-search engine (default)
└── Qdrant adapter (optional scaling backend)
          │
          ▼
Product grouping, filtering, and reranking
          │
          ▼
REST API, JavaScript client, and optional widget
          │
          ▼
Search analytics and relevance feedback
```

### Ownership boundaries

The project owns:

- Source connectors and synchronization
- Product and image normalization
- Embedding orchestration and model versioning
- Product-level search behavior
- Multi-image grouping
- Filtering and reranking
- Tenant isolation
- APIs and clients
- Analytics and evaluation
- The native exact-search engine

An optional backend such as Qdrant owns only vector persistence, nearest-neighbor candidate retrieval, and backend-specific filtering. It must not contain product-domain rules.

## 9. Main components

### API service

Responsibilities:

- Authentication and tenant resolution
- Source management
- Search requests
- Product lookup
- Feedback events
- Health and readiness endpoints
- Request validation and rate-limit hooks

### Worker

Responsibilities:

- Source synchronization
- Image downloading and validation
- Image normalization
- Embedding generation
- Batched indexing and deletion
- Retryable jobs and progress reporting

For the first release, the API and worker may run from the same application image with different commands.

### Metadata store

The default metadata store is SQLite. It stores tenants, sources, synchronization state, products, product images, model versions, jobs, and analytics events.

The SQLite database must use migrations and foreign keys. Database access must include tenant scope in repository methods rather than relying only on API-layer filtering.

### Embedding provider

The default provider runs a CLIP-compatible image/text embedding model locally. Model choice must be configurable. The repository must document the code license and model-weight license for the default model.

All stored vectors must record:

- Provider name
- Model identifier
- Model revision
- Vector dimension
- Normalization method
- Creation timestamp

Vectors produced by incompatible model versions must never be searched together.

### Native search engine

The native engine is required and enabled by default.

Version one uses exact cosine-similarity search. It may persist vectors as SQLite BLOBs or an explicitly versioned binary format and load them into a contiguous in-memory structure for searching.

The engine must support:

- Upsert by stable vector ID
- Delete by vector ID or product ID
- Tenant filtering
- Basic product metadata filters
- Exact top-K cosine similarity
- Multiple images per product
- Persistent data across restarts
- Rebuilding its in-memory state from durable storage
- Model-version separation

Approximate-nearest-neighbor indexing is not required for version one. HNSW or another native index may be added later without changing the public search API.

### Optional Qdrant adapter

Qdrant is an optional backend for larger catalogues. It must implement exactly the same `SearchEngine` contract as the native engine.

The default Docker Compose path must work without Qdrant. A Compose profile may enable it.

Do not describe the product as a new vector database. Describe it as a visual product-search platform with interchangeable search engines.

## 10. Canonical data model

### Tenant

```json
{
  "id": "tenant_01",
  "name": "Example Shop",
  "status": "active",
  "created_at": "2026-08-29T00:00:00Z"
}
```

### Source

```json
{
  "id": "src_01",
  "tenant_id": "tenant_01",
  "type": "local_manifest",
  "name": "Main catalogue",
  "config": {},
  "status": "ready",
  "sync_cursor": null,
  "last_synced_at": null
}
```

Secrets must not be returned through the API or stored directly inside general source configuration JSON.

### Product

```json
{
  "id": "prod_123",
  "tenant_id": "tenant_01",
  "source_id": "src_01",
  "external_id": "SHOE-123",
  "title": "Black Running Shoe",
  "description": "Lightweight running shoe",
  "category": "shoes",
  "brand": "Example",
  "price": 79.99,
  "currency": "USD",
  "in_stock": true,
  "attributes": {
    "color": "black",
    "gender": "unisex"
  },
  "source_url": "https://shop.example/products/SHOE-123",
  "content_hash": "sha256:...",
  "updated_at": "2026-08-29T00:00:00Z"
}
```

### Product image

```json
{
  "id": "img_456",
  "tenant_id": "tenant_01",
  "product_id": "prod_123",
  "source_uri": "file:///catalogue/SHOE-123-front.jpg",
  "position": 0,
  "content_hash": "sha256:...",
  "width": 1200,
  "height": 1200,
  "media_type": "image/jpeg",
  "embedding_status": "ready",
  "model_version": "provider:model@revision"
}
```

### Vector record

```text
vector_id
tenant_id
product_id
image_id
model_version
dimension
normalized
vector
created_at
```

### Search event

```json
{
  "id": "search_789",
  "tenant_id": "tenant_01",
  "search_type": "image",
  "filters": {"category": "shoes", "in_stock": true},
  "result_count": 20,
  "latency_ms": 84,
  "engine": "native",
  "model_version": "provider:model@revision",
  "created_at": "2026-08-29T00:00:00Z"
}
```

Raw shopper images must not be retained by default after query processing. Retention requires an explicit tenant setting and documented privacy behavior.

## 11. Connector contract

Every connector must expose equivalent behavior:

```python
class Connector:
    def validate(self) -> ValidationResult: ...
    def iter_changes(self, cursor: str | None) -> ChangePage: ...
    def fetch_product(self, external_id: str) -> SourceProduct: ...
    def open_image(self, image_ref: str) -> BinaryIO: ...
    def checkpoint(self) -> str | None: ...
```

`ChangePage` must distinguish `upsert` and `delete` operations. Connector results must use stable external IDs.

### First-release connector

Implement a local manifest connector first. It reads a JSON Lines manifest and images from a mounted directory.

Example manifest line:

```json
{"id":"SHOE-123","title":"Black Running Shoe","category":"shoes","price":79.99,"currency":"USD","in_stock":true,"images":["images/SHOE-123-front.jpg","images/SHOE-123-side.jpg"],"source_url":"https://shop.example/products/SHOE-123"}
```

After the local connector is reliable, add plain CSV/JSON, remote image URL, and S3-compatible connectors.

## 12. Search-engine contract

```python
class SearchEngine:
    def upsert(self, records: list[VectorRecord]) -> None: ...
    def delete(self, tenant_id: str, vector_ids: list[str]) -> None: ...
    def search(self, request: VectorSearchRequest) -> list[VectorHit]: ...
    def health(self) -> EngineHealth: ...
```

`VectorSearchRequest` must include:

- `tenant_id`
- `model_version`
- Query vector
- Candidate limit
- Supported metadata filters
- Excluded product IDs

The tenant ID must be mandatory and must never default to an all-tenant search.

## 13. Search and ranking algorithm

### Image preprocessing

At minimum:

- Decode the image rather than trusting its filename or MIME header.
- Apply EXIF orientation.
- Convert to the model's required color space.
- Reject unsupported or oversized images.
- Resize according to the embedding provider's specification.
- Apply an optional validated crop rectangle.

Automatic object detection and interactive object selection are Phase 2 features.

### Candidate retrieval

1. Generate a normalized query vector.
2. Request more image candidates than the final product limit.
3. Search only the active tenant and compatible model version.
4. Apply strict filters supported by the search engine.
5. Return image-level similarity scores.

For normalized vectors, cosine similarity is equivalent to their dot product:

```text
cosine_similarity(q, d) = (q · d) / (||q|| × ||d||)
```

### Product grouping

A product may contain several images. Group image hits by product and calculate an initial product score using the strongest image match:

```text
visual_score(product) = max(similarity(query, product_image_i))
```

Return the best matching image as `matched_image`. Do not return the same product multiple times.

### Reranking

The initial configurable formula is:

```text
final_score =
    0.70 × visual_score
  + 0.15 × text_or_attribute_score
  + 0.10 × category_compatibility
  + 0.05 × merchant_rule_score
```

Rules:

- Visual similarity remains the main signal.
- Stock, category, price, and tenant conditions may be strict filters rather than score adjustments.
- Every score component must be normalized to a documented range.
- The response may expose an overall score but must not expose misleading fake percentages.
- Ranking weights must be configurable per deployment before they become configurable per tenant.
- Future learning-to-rank work must be evaluated against a versioned relevance dataset.

## 14. REST API

All version-one endpoints use `/v1`.

### Required endpoints

```text
GET    /health
GET    /ready
POST   /v1/tenants
POST   /v1/sources
GET    /v1/sources
GET    /v1/sources/{source_id}
POST   /v1/sources/{source_id}/sync
GET    /v1/jobs/{job_id}
GET    /v1/products/{product_id}
POST   /v1/search/image
POST   /v1/search/products/{product_id}/similar
POST   /v1/search/{search_id}/feedback
```

### Image-search request

Use `multipart/form-data`:

```text
image: binary file
limit: integer, default 20, maximum configured by server
category: optional string
in_stock: optional boolean
min_price: optional decimal
max_price: optional decimal
crop_x: optional normalized number from 0 to 1
crop_y: optional normalized number from 0 to 1
crop_width: optional normalized number from 0 to 1
crop_height: optional normalized number from 0 to 1
```

### Search response

```json
{
  "search_id": "search_789",
  "took_ms": 84,
  "results": [
    {
      "product": {
        "id": "prod_123",
        "external_id": "SHOE-123",
        "title": "Black Running Shoe",
        "category": "shoes",
        "price": 79.99,
        "currency": "USD",
        "in_stock": true,
        "source_url": "https://shop.example/products/SHOE-123"
      },
      "matched_image": {
        "id": "img_456",
        "url": "/v1/products/prod_123/images/img_456"
      },
      "score": 0.873
    }
  ]
}
```

### Feedback request

```json
{
  "event": "click",
  "product_id": "prod_123",
  "position": 1
}
```

Supported initial events are `click`, `relevant`, and `not_relevant`.

## 15. Authentication and tenant isolation

The local development mode may use static API keys. Every key belongs to exactly one tenant unless it is an explicit administrative key.

Mandatory rules:

- Resolve tenant identity from authentication, not from an untrusted request body.
- Include tenant scope in every product, image, vector, source, job, and analytics query.
- Never perform an unscoped vector search.
- Prevent one tenant from referencing another tenant's product or source ID.
- Redact credentials and authorization headers from logs.
- Restrict outbound URL fetching to mitigate SSRF.
- Limit uploaded file size, decoded image dimensions, and processing time.
- Use constant-time comparison for static API keys where applicable.

## 16. Synchronization correctness

Use the following idempotency identity:

```text
tenant_id + source_id + external_product_id + content_hash
```

Image content hashes must be calculated from downloaded bytes. Product hashes must use a canonical serialization of searchable fields and ordered image references.

If an image changes while the external product ID remains the same, replace its vector. If a product is deleted from the source, remove its vectors and mark or delete its local product record according to the configured retention policy.

## 17. Docker developer experience

The required first-run flow is:

```bash
git clone <repository-url>
cd <repository>
docker compose up --build
```

After startup:

- API is reachable at `http://localhost:8080`.
- API documentation is reachable at `http://localhost:8080/docs`.
- Example catalogue data can be indexed without creating an external account.
- Persistent application data uses a named Docker volume.
- Example source files are mounted read-only.

Conceptual Compose structure:

```yaml
services:
  api:
    build: .
    command: ["serve"]
    ports:
      - "8080:8080"
    volumes:
      - app_data:/data
      - ./examples/catalogue:/sources/example:ro

  worker:
    build: .
    command: ["worker"]
    volumes:
      - app_data:/data
      - ./examples/catalogue:/sources/example:ro

  qdrant:
    image: qdrant/qdrant
    profiles: ["qdrant"]
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  app_data:
  qdrant_data:
```

The implementation may start with one application process if SQLite job claiming and concurrent writes are not yet safe across processes. Reliability is more important than showing multiple services prematurely.

## 18. Configuration

Initial environment variables:

```text
APP_ENV=development
APP_DATA_DIR=/data
APP_HOST=0.0.0.0
APP_PORT=8080
APP_API_KEY=<development-key>
SEARCH_ENGINE=native
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=<model-identifier>
MAX_UPLOAD_BYTES=10485760
MAX_IMAGE_PIXELS=40000000
QUERY_IMAGE_RETENTION=disabled
LOG_LEVEL=INFO
```

Optional Qdrant settings:

```text
SEARCH_ENGINE=qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
```

Validate configuration at startup and fail with actionable error messages.

## 19. Suggested repository structure

```text
open-visual-search/
├── app/
│   ├── api/
│   ├── auth/
│   ├── connectors/
│   ├── db/
│   ├── embeddings/
│   ├── images/
│   ├── jobs/
│   ├── models/
│   ├── ranking/
│   ├── search_engines/
│   │   ├── base.py
│   │   ├── native.py
│   │   └── qdrant.py
│   ├── services/
│   └── settings.py
├── clients/
│   └── javascript/
├── docs/
│   ├── architecture.md
│   ├── connectors.md
│   ├── search-quality.md
│   └── security.md
├── examples/
│   └── catalogue/
├── migrations/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── relevance/
├── Dockerfile
├── compose.yaml
├── LICENSE
├── README.md
└── pyproject.toml
```

## 20. Observability and analytics

Record:

- Synchronization duration and outcome
- Products and images discovered, changed, skipped, failed, and deleted
- Embedding latency and failures
- Search latency by processing stage
- Candidate and final result counts
- Zero-result searches
- Click and relevance feedback
- Engine and model version

Do not log raw API keys, source credentials, complete uploaded image bytes, or sensitive source configuration.

## 21. Search-quality evaluation

Create a small versioned evaluation catalogue with permitted images and expected matches.

Measure at least:

- Recall@K: whether a relevant product appears in the top K
- Precision@K: how many top-K products are relevant
- Mean reciprocal rank: how early the first relevant product appears
- Zero-result rate
- Search latency percentiles

Every model or ranking change must be compared with the previous version on the same dataset. Do not accept a relevance change based only on a few hand-picked examples.

## 22. Testing requirements

### Unit tests

- Image validation and crop calculations
- Canonical hashes
- Connector normalization
- Cosine similarity and top-K behavior
- Product grouping
- Ranking normalization
- Filter validation
- Tenant-scoped repository behavior

### Integration tests

- Index an example catalogue and search it
- Update and delete products through a second synchronization
- Restart the application and preserve results
- Reject cross-tenant product access
- Reject incompatible model versions
- Run the same engine contract tests against native and optional Qdrant adapters

### Security tests

- Malformed and decompression-bomb-like image inputs
- Unauthorized requests
- Cross-tenant IDs
- Source paths escaping their allowed root
- Unsafe remote URLs
- Oversized uploads

## 23. Version-one acceptance criteria

Version one is complete only when all of the following are true:

- `docker compose up --build` starts the platform without Qdrant.
- A developer can index the included local example catalogue.
- Image search returns product-level results rather than duplicate image rows.
- Results include the best matching product image and original source URL.
- Category, stock, and price filters work.
- Similar-product search works using an indexed product ID.
- Index data survives a container restart.
- Changed products are re-indexed and deleted products disappear from search.
- Two test tenants cannot access or discover each other's products.
- The native engine passes the engine contract test suite.
- Search and synchronization errors are actionable.
- The repository documents the default model's licenses and attribution.
- No paid account or external hosted service is required.

## 24. Delivery roadmap

### Milestone 0: foundation

- Repository, license, formatting, tests, and continuous integration
- Configuration and structured logging
- SQLite schema and migrations
- Tenant-scoped authentication
- Core domain models and interfaces

### Milestone 1: end-to-end native search

- Local manifest connector
- Image validation and normalization
- Local embedding provider
- Native exact vector engine
- Catalogue synchronization
- Image and similar-product endpoints
- Example catalogue and Docker Compose

### Milestone 2: product-quality search

- Product grouping and configurable reranking
- Category, price, stock, and attribute filters
- Search analytics and feedback
- Relevance evaluation suite
- Optional crop selection and object detection

### Milestone 3: integrations and scale

- Qdrant adapter
- URL and S3-compatible connectors
- JavaScript client and embeddable widget
- PostgreSQL metadata option if needed
- Operational documentation and backup procedures

### Milestone 4: advanced relevance

- Text-to-product and combined text-plus-image search
- Domain-specific reranking
- Duplicate and near-duplicate controls
- Configurable multi-vector product representations
- Carefully evaluated learning-to-rank support

## 25. Instructions for coding agents

An implementation agent must:

1. Read this entire specification before changing code.
2. Implement the smallest current milestone end to end.
3. Keep domain logic independent from FastAPI, SQLite, and Qdrant adapters.
4. Keep the native engine as the default and ensure it works without Qdrant.
5. Put tenant scope in storage and search interfaces, not only request handlers.
6. Write tests for synchronization, deletion, persistence, and tenant isolation.
7. Use stable IDs and idempotent upserts.
8. Record embedding model versions with every vector.
9. Avoid adding infrastructure until a measured requirement justifies it.
10. Update documentation when behavior or interfaces change.

An implementation agent must not:

- Add a mandatory paid API or hosted service.
- Make Qdrant required for the default trial.
- Call the product a vector database.
- Mix vectors from different model versions.
- Retain shopper query images by default.
- Implement an unscoped cross-tenant search.
- Hide failed product or image indexing.
- Add automatic object detection before the basic end-to-end search is reliable.
- Optimize approximate search before the exact native engine is measured.

When a requirement is ambiguous, prefer a secure, local, reversible implementation and document the assumption.

## 26. Open decisions

These decisions must be resolved before or during Milestone 1 and documented:

- Final project and package name
- Programming language and supported runtime version
- Default local embedding model and its licenses
- SQLite vector persistence format
- Background-job strategy for the single-node release
- API-key creation and secure storage format
- Maximum supported upload size and image dimensions
- Initial open-source license for this repository

## 27. References

- [Qdrant documentation](https://qdrant.tech/documentation/)
- [Qdrant hybrid and multi-stage search](https://qdrant.tech/documentation/search/hybrid-queries/)
- [Qdrant Apache 2.0 repository](https://github.com/qdrant/qdrant)
- [Sentence Transformers documentation](https://sbert.net/)

These are implementation references, not mandatory hosted services. Any dependency or model added to the project must have its license reviewed and recorded.
