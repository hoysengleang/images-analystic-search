from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.providers.qdrant_provider import check_qdrant_connection

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    app_name: str
    status: str
    qdrant: Optional[dict[str, Optional[str]]] = None


@router.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_none=True,
    summary="Health check",
)
def health_check(
    include_qdrant: bool = Query(
        default=False,
        description="Include Qdrant connection status in the health response.",
    ),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    qdrant_status = None
    if include_qdrant:
        qdrant_status = check_qdrant_connection(settings=settings).to_dict()

    return HealthResponse(
        app_name=settings.app_name,
        status="ok",
        qdrant=qdrant_status,
    )
