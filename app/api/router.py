from fastapi import APIRouter, Depends

from app.api.routes.collections import router as collections_router
from app.api.routes.health import router as health_router
from app.api.routes.images import router as images_router
from app.api.routes.index import router as index_router
from app.api.routes.models import router as models_router
from app.api.routes.search import router as search_router
from app.core.security import require_api_key

api_router = APIRouter()
api_router.include_router(health_router)

protected_router = APIRouter(dependencies=[Depends(require_api_key)])
protected_router.include_router(models_router)
protected_router.include_router(collections_router)
protected_router.include_router(index_router)
protected_router.include_router(search_router)
protected_router.include_router(images_router)

api_router.include_router(protected_router)
