# OpenVisionSearch

[![CI](https://github.com/hoysengleang/images-analystic-search/actions/workflows/ci.yml/badge.svg)](https://github.com/hoysengleang/images-analystic-search/actions/workflows/ci.yml)
[![CodeQL](https://github.com/hoysengleang/images-analystic-search/actions/workflows/codeql.yml/badge.svg)](https://github.com/hoysengleang/images-analystic-search/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9–3.12](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue.svg)](pyproject.toml)

OpenVisionSearch is a free, open-source, self-hosted **visual product search
platform**. Its current, pre-1.0 image-level API indexes images from URLs,
uploads, local paths, folders, base64 payloads, or S3/R2 signed URLs and searches
them over plain HTTP from any language. The product-level `/v1` API is the next
milestone.

Search by **image**, by **text**, or by both at once. It stores **vectors and metadata, not your original images**.

```text
image → OpenCLIP encoder → vector → Qdrant nearest-neighbour search → similar images
```

> **Where this is heading.** The endpoints below are image-level and stable to
> build against today. The project is being extended into a multi-tenant
> **product** search platform — one result per product rather than per image,
> catalogue connectors, and a native engine that removes the Qdrant
> requirement. The storage and tenancy foundation for that is built and tested
> but not yet wired to the API. The
> [product specification](docs/VISUAL_PRODUCT_SEARCH_SPEC.md) is the source of
> truth; see the [roadmap](docs/roadmap.md) and [decision records](docs/adr/) for
> implementation status and deviations.

## Quickstart

```bash
docker compose up --build
```

Put images you want to index under `./data/images` on the host; the container sees them at `/data/images`.

Open the interactive docs at http://localhost:8000/docs.

### 60-second walkthrough

Create a collection:

```bash
curl -X POST http://localhost:8000/collections -H 'Content-Type: application/json' -d '{"name":"products"}'
```

Index a few images:

```bash
curl -X POST http://localhost:8000/collections/products/index -H 'Content-Type: application/json' -d '{"images":[{"id":"shoe_001","source":{"type":"url","value":"https://cdn.example.com/shoe.jpg"},"metadata":{"name":"Blue Shoe","category":"shoes"}}]}'
```

Search with a photo from a phone:

```bash
curl -X POST http://localhost:8000/collections/products/search/upload -F 'image=@query.jpg' -F 'top_k=10'
```

Or search the same images with words, no re-indexing needed:

```bash
curl -X POST http://localhost:8000/collections/products/search/text -H 'Content-Type: application/json' -d '{"query":"blue running shoe","top_k":10}'
```

More runnable samples live in [`examples/`](examples/) for curl, Python, JavaScript, and PHP — including [a tour of every endpoint](examples/curl/full-api-tour.sh).

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness; `?include_qdrant=true` also probes Qdrant |
| `GET` | `/models` | Embedding models this server can use |
| `POST` | `/collections` | Create a collection (pins its embedding model) |
| `GET` | `/collections` | List collections |
| `GET` | `/collections/{name}/stats` | Vector count, size, distance, model |
| `DELETE` | `/collections/{name}` | Delete a collection and its vectors |
| `POST` | `/collections/{name}/index` | Index images by URL, path, or base64 |
| `POST` | `/collections/{name}/index/upload` | Index one uploaded file |
| `POST` | `/collections/{name}/index/folder` | Index every image in a mounted folder |
| `POST` | `/collections/{name}/search` | Search by URL, path, or base64 |
| `POST` | `/collections/{name}/search/upload` | Search by uploaded file |
| `POST` | `/collections/{name}/search/text` | Search with a text description |
| `POST` | `/collections/{name}/search/hybrid` | Search by image, steered by text |
| `POST` | `/collections/{name}/search/batch` | Search with several query images |
| `GET` | `/collections/{name}/images` | Page through indexed records |
| `GET` | `/collections/{name}/images/{id}` | Read one record |
| `DELETE` | `/collections/{name}/images/{id}` | Delete one record |

Full request and response shapes: [`docs/api-reference.md`](docs/api-reference.md).

## Image sources

| Source | How you send it |
| --- | --- |
| Public URL / CDN | `{"type":"url","value":"https://…"}` |
| S3, R2, MinIO, Spaces, Supabase | Generate a **signed URL** in your backend and send it as a `url` source |
| Upload | `multipart/form-data` to the `/index/upload` and `/search/upload` endpoints |
| Local path | `{"type":"path","value":"/data/images/a.jpg"}` — a path inside the container |
| Folder | `POST /collections/{name}/index/folder` with a folder under `ALLOWED_IMAGE_ROOT` |
| Base64 | `{"type":"base64","value":"/9j/4AAQ…"}` — no `data:` prefix |

Details and private-storage guidance: [`docs/image-sources.md`](docs/image-sources.md) and [`docs/storage-strategy.md`](docs/storage-strategy.md).

## Configuration

Copy the template and edit what you need:

```bash
cp .env.example .env
```

| Variable | Default | Description |
| --- | --- | --- |
| `APP_NAME` | `OpenVisionSearch` | Name shown in the docs and health response |
| `APP_ENV` | `development` | Environment label; read by Compose, not by the app |
| `APP_VERSION` | `0.1.0` | Version reported in the docs and OpenAPI schema |
| `APP_HOST` / `APP_PORT` | `0.0.0.0` / `8000` | Passed to uvicorn by the container command |
| `APP_DATA_DIR` | `data` | Where SQLite and other durable state live |
| `LOG_LEVEL` | `INFO` | Root log level |
| `LOG_JSON` | `true` | Structured JSON logs with credentials redacted |
| `SEARCH_ENGINE` | `qdrant` | Only `qdrant` today; the native engine arrives in Milestone 1 |
| `API_PREFIX` | empty | Prefix applied to every route |
| `API_KEY` | empty | When set, every route except `/health` requires `X-API-Key` |
| `CORS_ALLOW_ORIGINS` | `["*"]` | Browser origins allowed to call the API |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant endpoint |
| `QDRANT_API_KEY` | empty | Qdrant API key, if your instance needs one |
| `COLLECTION_METADATA_PATH` | `data/collections.json` | Where the collection→model registry is kept |
| `DEFAULT_EMBEDDING_PROVIDER` | `openclip` | Provider used when a collection does not name one |
| `DEFAULT_MODEL_NAME` | `ViT-B-32` | Default OpenCLIP architecture |
| `DEFAULT_MODEL_PRETRAINED` | `laion2b_s34b_b79k` | Default pretrained weights |
| `DEFAULT_VECTOR_SIZE` | `512` | Must match the model's output size |
| `DEFAULT_DISTANCE` | `cosine` | Distance metric for new collections |
| `IMAGE_FRAMING` | `pad` | `pad` keeps the whole photo; `crop` is the stock CLIP transform |
| `EMBED_VIEWS` | `1` | Views averaged per image; `3` trades speed for accuracy |
| `EMBED_BATCH_SIZE` | `16` | Images encoded per forward pass |
| `WARMUP_MODEL_ON_STARTUP` | `false` | Load model weights at boot instead of on first request |
| `ONNX_IMAGE_MODEL_PATH` | empty | Exported image encoder; required by the `onnx` provider |
| `ONNX_TEXT_MODEL_PATH` | empty | Exported text encoder; without it the `onnx` provider is image-only |
| `ONNX_TOKENIZER_PATH` | empty | The `tokenizer.json` that text encoder was exported with |
| `DETECTOR_PROVIDER` | `none` | Query-side object detector; `owlvit` enables `detect` |
| `DETECTOR_MODEL_PATH` | empty | Exported detector, required when `DETECTOR_PROVIDER` is set |
| `DETECTOR_TOKENIZER_PATH` | empty | The `tokenizer.json` that detector was exported with |
| `DETECTOR_PROMPTS` | `["a photo of an object"]` | What to look for when the caller does not say |
| `DETECTOR_MIN_SCORE` | `0.05` | Detector confidence below which a region is ignored |
| `MAX_QUERY_REGIONS` | `1` | Most regions searched per query; each costs a round trip |
| `ALLOWED_IMAGE_ROOT` | `/data/images` | Only paths inside this root can be read |
| `MAX_IMAGE_SIZE_MB` | `20` | Hard cap for uploads, downloads, and base64 |
| `MAX_IMAGE_PIXELS` | `40000000` | Decoded width x height ceiling, guarding decompression bombs |
| `MAX_REQUEST_BODY_MB` | `25` | Whole-request cap, enforced before the body is read |
| `ALLOWED_IMAGE_TYPES` | jpeg, png, webp | Accepted image MIME types |
| `URL_DOWNLOAD_TIMEOUT_SECONDS` | `10` | Timeout per image download |
| `MAX_REDIRECTS` | `3` | Redirects followed when downloading |
| `ALLOW_PRIVATE_URLS` | `false` | Allow private/internal hosts (leave off in production) |
| `EMBED_BATCH_SIZE` | `16` | Also sets peak indexing memory: batch size x image size |
| `DEFAULT_TOP_K` | `10` | Results returned when the caller does not ask |
| `MAX_TOP_K` | `100` | Largest `top_k` a caller may request |
| `SEARCH_CANDIDATE_MULTIPLIER` | `4` | Candidates fetched per region before duplicates are folded away |

## Why results are better than a stock CLIP setup

The usual way to wire CLIP up resizes an image's short side and then centre-crops it. On a portrait phone photo that **throws away about 40% of the frame**, including whatever sat at the top and bottom — often part of the product itself.

`IMAGE_FRAMING=pad` (the default) scales the whole image to fit and pads the remainder, so the model sees everything the shopper photographed. Measured on 300 portrait products with degraded query photos:

| Framing | Views | Recall@1 | Recall@5 | Index | Query |
| --- | --- | --- | --- | --- | --- |
| `crop` (stock CLIP) | 1 | 75.0% | 96.2% | 7 ms | 17 ms |
| **`pad`** | **1** | **90.0%** | **100%** | 6 ms | 16 ms |
| `pad` | 3 | **95.0%** | 100% | 16 ms | 25 ms |

Padding costs nothing — it is fractionally *faster*, since there is less image to resample — and on already-square images it is a no-op producing byte-identical results. `EMBED_VIEWS=3` averages the frame with a centre crop and a mirror for another 5 points, at roughly double the indexing work.

Both settings change the vectors, so each collection stores the framing it was built with and keeps using it. Changing the server default does not disturb collections already indexed; to adopt a new setting, create a new collection and re-index.

## Searching by text

OpenCLIP encodes images and text into the same vector space, so the images you already indexed are searchable by description with no extra work and no second index:

```bash
# words only
curl -X POST http://localhost:8000/collections/products/search/text \
  -H 'Content-Type: application/json' \
  -d '{"query":"red running shoe with a white sole"}'

# an image, nudged by words: "this shoe, but blue"
curl -X POST http://localhost:8000/collections/products/search/hybrid \
  -H 'Content-Type: application/json' \
  -d '{"source":{"type":"url","value":"https://cdn.example.com/shoe.jpg"},"query":"but in blue","text_weight":0.3}'
```

Text-to-image scores sit lower than image-to-image scores (roughly 0.2–0.35 for a good match, versus 0.8+), so tune `min_score` per endpoint rather than reusing one threshold. `GET /models` reports `supports_text` for each model.

## One collection, one model

Vectors from different models are not comparable, so a collection is pinned to the embedding model it was created with. That model is recorded in the collection registry and reused for every later index and search call. To switch models, create a new collection and re-index.

## Searching part of a picture

A shopper's photo has a background. Embedding the whole frame gives you a vector
describing the table as much as the bag on it, and the match suffers. Two ways to
narrow it, both query-side — your indexed images are untouched either way.

**Tell the server where to look.** Normalized coordinates, so the same box works
for a thumbnail and the original:

```bash
curl -X POST http://localhost:8000/collections/products/search \
  -H 'Content-Type: application/json' \
  -d '{"source":{"type":"url","value":"https://example.com/photo.jpg"},"crop":{"x":0.31,"y":0.44,"width":0.22,"height":0.30}}'
```

On the upload endpoint the same box is four form fields — `crop_x`, `crop_y`,
`crop_width`, `crop_height` — because multipart has nowhere to put an object.

**Or let the server find the objects itself.** Off by default; see
[the detector setup](#running-without-pytorch) below for the model files:

```bash
curl -X POST http://localhost:8000/collections/products/search \
  -H 'Content-Type: application/json' \
  -d '{"source":{"type":"url","value":"https://example.com/photo.jpg"},"detect":true,"detect_prompt":["a handbag"]}'
```

Each region found is searched, and an image keeps its best score across them.
The response reports what was searched, so you can draw the boxes:

```json
{
  "results": [{"id": "SKU-001", "score": 0.91, "matched_query_region": 0}],
  "query_regions": [
    {"box": {"x": 0.13, "y": 0.23, "width": 0.25, "height": 0.49},
     "score": 0.73, "label": "a handbag"}
  ]
}
```

Three things worth knowing:

- **Name what you are looking for.** The detector is open-vocabulary, so it has
  no class list and scores whatever you ask for. On one cluttered test scene
  `"a handbag"` scored 0.73, the default generic prompt scored 0.16, and a bare
  `"an object"` found nothing at all. Pass `detect_prompt`.
- **Finding nothing is not an error.** The search falls back to the whole image
  and returns an empty `query_regions`, rather than refusing to answer.
- **More regions is not better.** Scores fold by taking each product's best
  region, so an extra weak region can only lift whatever it resembles. On a bag
  photographed beside a laptop, searching the strongest region put the bag 0.257
  clear of the runner-up; adding a second, low-confidence region lifted the
  laptop and cut that lead to 0.073 — worse than not detecting at all. Hence
  `MAX_QUERY_REGIONS=1`. Raise it only if you want every object in the picture.
- **It is not free.** The detector costs about 200 ms per query on CPU against
  about 20 ms to embed an image, so a search goes from roughly 25 ms to roughly
  220 ms. An explicit `crop` gives the same accuracy for the same 25 ms.
- **Measure before you trust it.** Detection has not been shown to improve
  relevance on real catalogues — only that it finds the right box. Run
  `benchmarks/relevance.py` on your own images with `detect` off and on before
  turning it on in production.

Cluttered *catalogue* images are a separate problem, and not solved here: each
indexed image is still one vector. See
[ADR 0007](docs/adr/0007-query-side-region-search.md).

## Running without PyTorch

The default `openclip` provider pulls in PyTorch, which dominates the image
size. If you are deploying somewhere small, export the model once and serve it
with ONNX Runtime instead: identical vectors, a much smaller dependency.

The saving is in the libraries, not the weights — the exported model is about
the size of the checkpoint it came from. As a rough gauge, on one macOS/Python
3.9 environment `torch` plus `torchvision` measured ~346 MB against ~68 MB for
`onnxruntime` plus `tokenizers`; measure your own target platform, since the
Linux container wheels differ.

Export on any machine that can install the full requirements:

```bash
PYTHONPATH=. python scripts/export_onnx.py --output-dir data/models
```

That writes an image encoder, a text encoder, and the tokenizer, then loads all
three back through the provider that will serve them and compares the result
against PyTorch on real inputs. It refuses to finish if they disagree, so an
export that would quietly degrade search never reaches a server.

Copy the three files across, install the smaller runtime, and point the service
at them:

```bash
pip install -r requirements-onnx.txt
```

```bash
DEFAULT_EMBEDDING_PROVIDER=onnx
ONNX_IMAGE_MODEL_PATH=data/models/image_encoder.onnx
ONNX_TEXT_MODEL_PATH=data/models/text_encoder.onnx
ONNX_TOKENIZER_PATH=data/models/tokenizer.json
```

To enable the query-side detector, export it the same way — it runs on the same
ONNX runtime, so nothing else is needed:

```bash
PYTHONPATH=. python scripts/export_detector_onnx.py --output-dir data/models
```

```bash
DETECTOR_PROVIDER=owlvit
DETECTOR_MODEL_PATH=data/models/detector.onnx
DETECTOR_TOKENIZER_PATH=data/models/detector_tokenizer.json
```

That export is verified too: the tensor this repository builds from an image is
compared against the reference processor's, and the exported graph's outputs
against PyTorch's, before the files are accepted.

Leave `ONNX_TEXT_MODEL_PATH` unset if you only ever search by image. `/models`
then reports `supports_text: false`, and text and hybrid search are refused
rather than answered out of the wrong vector space.

Framing, multi-view averaging, and batching behave exactly as they do under
`openclip`: the preprocessing is a byte-for-byte reimplementation of the same
transform, and a test holds it to that.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

```bash
python -m pytest
```

## Measuring search quality

Accuracy is a property of your catalogue, not of the code, so measure it on your own images:

```bash
PYTHONPATH=. python benchmarks/relevance.py --image-dir ./data/images --catalogue 5000 --queries 200
```

It indexes the catalogue, queries with degraded copies of each image the way a phone photo differs from a catalogue shot, and reports Recall@1/@5/@10, MRR, and per-stage latency. Run it before and after any model or ranking change; never accept a relevance change from a handful of hand-picked examples.

Two things to keep in mind: a Recall@1 figure is only meaningful alongside the catalogue size it was measured on (more images means more chances to confuse), and the built-in synthetic catalogue exercises the pipeline but is **not** predictive of real product photos.

## Documentation

- [Product specification — source of truth](docs/VISUAL_PRODUCT_SEARCH_SPEC.md)
- [Architecture](docs/architecture.md)
- [API reference](docs/api-reference.md)
- [Image sources](docs/image-sources.md)
- [Storage strategy](docs/storage-strategy.md)
- [Security model](docs/security.md)
- [Deployment](docs/deployment.md)
- [Roadmap](docs/roadmap.md)
- [Contributing](CONTRIBUTING.md), [support](SUPPORT.md), and
  [security reporting](SECURITY.md)

## Model licensing and attribution

OpenVisionSearch does not redistribute model weights. The default configuration
downloads `ViT-B-32` / `laion2b_s34b_b79k` through OpenCLIP at runtime.

- [OpenCLIP](https://github.com/mlfoundations/open_clip) and its source code are
  distributed under the [MIT License](https://github.com/mlfoundations/open_clip/blob/main/LICENSE).
- The default [LAION ViT-B/32 model card](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)
  labels the checkpoint MIT and documents its LAION-2B training data, intended
  uses, limitations, and citation information. Review that model card before
  deploying the model in your own domain.

## License

MIT — see [LICENSE](LICENSE).
