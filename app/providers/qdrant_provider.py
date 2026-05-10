from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional, TYPE_CHECKING

from app.core.config import Settings, get_settings

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


@dataclass(frozen=True)
class QdrantConnectionStatus:
    status: str
    url: str
    version: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Optional[str]]:
        return asdict(self)


def create_qdrant_client(settings: Optional[Settings] = None) -> "QdrantClient":
    from qdrant_client import QdrantClient

    settings = settings or get_settings()

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=5,
    )


@lru_cache
def get_qdrant_client() -> "QdrantClient":
    return create_qdrant_client()


def check_qdrant_connection(
    client: Optional["QdrantClient"] = None,
    settings: Optional[Settings] = None,
) -> QdrantConnectionStatus:
    settings = settings or get_settings()
    client = client or get_qdrant_client()

    try:
        info = client.info()
    except Exception as exc:
        return QdrantConnectionStatus(
            status="unavailable",
            url=settings.qdrant_url,
            error=str(exc),
        )

    return QdrantConnectionStatus(
        status="ok",
        url=settings.qdrant_url,
        version=getattr(info, "version", None),
    )
