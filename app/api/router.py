from fastapi import APIRouter

from app.api.routes.collections import router as collections_router
from app.api.routes.health import router as health_router
from app.api.routes.models import router as models_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(models_router)
api_router.include_router(collections_router)
