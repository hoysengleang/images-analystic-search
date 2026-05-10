# OpenVisionSearch

OpenVisionSearch is a free, open-source, self-hosted visual search API. Developers will be able to index images from URLs, uploads, local paths, folders, base64 payloads, signed S3 URLs, or other storage services, then search for similar images through simple REST endpoints.

This repository currently contains the initial FastAPI project foundation. Qdrant vector search and OpenCLIP image embeddings are intentionally not implemented yet.

## Current Status

- FastAPI application shell
- Central API router
- Environment-based configuration
- Health endpoint
- Docker and Docker Compose setup
- Qdrant service included in Compose for future integration

## Project Principles

- Store vectors and metadata by default, not original images.
- Keep embedding model metadata at the collection level because vectors from different models cannot safely mix.
- Design service boundaries so future users can choose different embedding models.
- Expose language-independent REST APIs that are easy to call from any backend.

## Quickstart

Create a local environment file if you want to override defaults:

```bash
cp .env.example .env
```

Run the stack:

```bash
docker compose up --build
```

The API will be available at:

- Health: [http://localhost:8000/health](http://localhost:8000/health)
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- OpenAPI JSON: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

Expected health response:

```json
{
  "app_name": "OpenVisionSearch",
  "status": "ok"
}
```

## Local Development

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app.main:app --reload
```

## Configuration

Configuration is loaded from environment variables and, when present, a local `.env` file.

| Variable | Default | Description |
| --- | --- | --- |
| `APP_NAME` | `OpenVisionSearch` | Name returned by the API and shown in docs |
| `APP_VERSION` | `0.1.0` | Application version |
| `ENVIRONMENT` | `development` | Runtime environment label |
| `API_PREFIX` | empty | Optional prefix for all API routes |
| `QDRANT_URL` | `http://qdrant:6333` | Reserved for future Qdrant integration |

## API

### `GET /health`

Returns basic service health and the configured application name.

## Planned Structure

Future work should keep integrations behind service boundaries:

```text
app/
  api/
    routes/
  core/
  models/
  schemas/
  services/
    embeddings/
    vector_store/
    image_sources/
```

The first embedding service will use OpenCLIP. The first vector store service will use Qdrant.
