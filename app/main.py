from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestBodySizeLimitMiddleware
from app.dependencies import get_embedding_manager, get_image_loader

#: Pinned so the docs page cannot change under a running deployment, the same
#: reason the Compose file pins its images. Served from a CDN, as FastAPI's own
#: Swagger UI already was, so an air-gapped install still needs to vendor it.
SCALAR_SCRIPT_URL = (
    "https://cdn.jsdelivr.net/npm/@scalar/api-reference@1.67.0/dist/browser/standalone.js"
)

DESCRIPTION = """
Self-hosted visual search API.

Index images from URLs, uploads, local paths, folders, or base64 payloads,
then search for visually similar images from any language over plain HTTP.

Vectors and metadata are stored; original images are not.
"""


def register_api_reference(app: FastAPI, *, title: str) -> None:
    """Serve the OpenAPI schema through Scalar at ``/docs``.

    Scalar renders the same schema as Swagger UI did, with a request console
    that is usable enough that a developer can try the API before writing any
    client code. It is one script tag over the schema FastAPI already emits.
    """
    page = f"""<!doctype html>
<html>
  <head>
    <title>{title} API</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
  </head>
  <body>
    <div id="app"></div>
    <script src="{SCALAR_SCRIPT_URL}"></script>
    <script>
      Scalar.createApiReference('#app', {{ url: '{app.openapi_url}' }})
    </script>
  </body>
</html>"""

    @app.get("/docs", include_in_schema=False)
    def api_reference() -> HTMLResponse:
        return HTMLResponse(page)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.warmup_model_on_startup:
        get_embedding_manager().get_default_provider().warmup()
    try:
        yield
    finally:
        get_image_loader().close()


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    explicit_settings = settings
    settings = settings or get_settings()

    configure_logging(level=settings.log_level, json_output=settings.log_json)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=DESCRIPTION,
        # Swagger UI's default mount is replaced by Scalar below; ReDoc stays
        # for readers who prefer a plain three-column reference.
        docs_url=None,
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    register_api_reference(app, title=settings.app_name)

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_middleware(
        RequestBodySizeLimitMiddleware,
        max_body_bytes=settings.max_request_body_bytes,
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)

    if explicit_settings is not None:
        # Without this, routes would still resolve Depends(get_settings) to the
        # process-wide settings and quietly ignore what the caller passed.
        app.dependency_overrides[get_settings] = lambda: explicit_settings

    return app


app = create_app()
