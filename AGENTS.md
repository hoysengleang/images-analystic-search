You are working on OpenVisionSearch.

Goal:
Build a free open-source FastAPI service for image similarity search. Developers can self-host it, index images from URL, upload, local path, folder, base64, or signed S3/third-party URLs, then search similar images through REST API endpoints.

Important:

- Use FastAPI.
- Use Qdrant for vector search.
- Use OpenCLIP as the first embedding model.
- Do not store original images by default.
- Store vectors and metadata only.
- Design the code so future users can select different embedding models.
- Keep collection-level model metadata because vectors from different models cannot mix.
- Make the API developer-friendly and language-independent.
