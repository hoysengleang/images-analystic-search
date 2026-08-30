"""Liveness endpoint, plus an opt-in Qdrant connectivity probe."""

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.core.config import Settings, get_settings
from app.dependencies import get_qdrant_client
from app.providers.qdrant_provider import check_qdrant_connection
from app.schemas.system import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_none=True,
    summary="Health check",
)
def health_check(
    include_qdrant: bool = Query(
        default=False,
        description="Also report whether Qdrant is reachable.",
    ),
    settings: Settings = Depends(get_settings),
    qdrant_client: Any = Depends(get_qdrant_client),
) -> HealthResponse:
    qdrant_status = None
    if include_qdrant:
        qdrant_status = check_qdrant_connection(
            client=qdrant_client,
            settings=settings,
        ).to_dict()

    return HealthResponse(
        app_name=settings.app_name,
        status="ok",
        search_engine=settings.search_engine,
        qdrant=qdrant_status,
    )
