"""API key authentication.

Authentication is opt-in: when ``API_KEY`` is unset the service stays open,
which keeps the self-hosted quickstart friction-free. As soon as ``API_KEY``
is configured, every route that depends on :func:`require_api_key` demands a
matching ``X-API-Key`` header.
"""

from __future__ import annotations

from secrets import compare_digest
from typing import Optional

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings
from app.core.errors import UnauthorizedError

API_KEY_HEADER_NAME = "X-API-Key"

api_key_header = APIKeyHeader(
    name=API_KEY_HEADER_NAME,
    auto_error=False,
    description="Required only when the server is started with API_KEY set.",
)


def require_api_key(
    provided_key: Optional[str] = Security(api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """Reject the request when the configured API key does not match."""
    expected_key = settings.api_key
    if expected_key is None:
        return

    if provided_key is None or not compare_digest(provided_key, expected_key):
        raise UnauthorizedError(
            message=f"A valid {API_KEY_HEADER_NAME} header is required",
            details={"header": API_KEY_HEADER_NAME},
        )
