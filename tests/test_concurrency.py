"""Concurrent traffic must not corrupt state or surface as server errors."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.dependencies import (
    get_collection_service,
    get_embedding_manager,
    get_indexing_service,
    get_qdrant_client,
    get_search_service,
    get_vector_service,
)
from app.embedding.base import EmbeddingProvider
from app.embedding.manager import EmbeddingManager
from app.embedding.registry import EmbeddingProviderRegistry
from app.main import create_app
from app.services.collection_metadata_service import CollectionMetadataService
from app.services.collection_service import CollectionService
from app.services.image_loader import ImageLoader
from app.services.indexing_service import IndexingService
from app.services.search_service import SearchService
from app.services.vector_service import VectorService


def unit(channels) -> list:
    norm = sum(v * v for v in channels) ** 0.5 or 1.0
    return [v / norm for v in channels]


class ColorProvider(EmbeddingProvider):
    provider_name = "color"

    def embed_image(self, image) -> list:
        pixels = list(image.convert("RGB").getdata())
        return unit([sum(p[c] for p in pixels) / len(pixels) for c in range(3)])


@pytest.fixture
def api(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    for index in range(12):
        buffer = BytesIO()
        Image.new("RGB", (16, 16), (20 * index % 256, 90, 140)).save(buffer, format="PNG")
        (image_root / f"p{index}.png").write_bytes(buffer.getvalue())

    settings = Settings(
        allowed_image_root=image_root,
        default_embedding_provider="color",
        default_model_name="color-v1",
        default_model_pretrained="none",
        default_vector_size=3,
    )
    registry = EmbeddingProviderRegistry()
    registry.register(ColorProvider)
    manager = EmbeddingManager(settings=settings, registry=registry)
    qdrant_client = QdrantClient(location=":memory:")
    metadata_service = CollectionMetadataService(tmp_path / "collections.json")
    vector_service = VectorService(qdrant_client=qdrant_client)
    services = {
        "settings": settings,
        "metadata_service": metadata_service,
        "image_loader": ImageLoader(settings=settings),
        "embedding_manager": manager,
        "vector_service": vector_service,
    }

    app = create_app(settings)
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant_client
    app.dependency_overrides[get_embedding_manager] = lambda: manager
    app.dependency_overrides[get_vector_service] = lambda: vector_service
    app.dependency_overrides[get_collection_service] = lambda: CollectionService(
        settings=settings,
        qdrant_client=qdrant_client,
        metadata_service=metadata_service,
        embedding_manager=manager,
    )
    app.dependency_overrides[get_indexing_service] = lambda: IndexingService(**services)
    app.dependency_overrides[get_search_service] = lambda: SearchService(**services)

    client = TestClient(app)
    client.image_root = image_root
    return client


def test_concurrent_collection_creation_yields_one_winner(api) -> None:
    """Ten racing creates: exactly one 201, the rest a clean conflict."""
    with ThreadPoolExecutor(max_workers=10) as pool:
        statuses = [
            future.status_code
            for future in pool.map(
                lambda _: api.post("/collections", json={"name": "race"}), range(10)
            )
        ]

    assert statuses.count(201) == 1
    assert all(status in {201, 409} for status in statuses)
    assert len(api.get("/collections").json()["collections"]) == 1


def test_concurrent_indexing_stores_every_image_once(api) -> None:
    api.post("/collections", json={"name": "c"})

    def index(index_number: int):
        return api.post(
            "/collections/c/index",
            json={
                "images": [
                    {
                        "id": f"p{index_number}",
                        "source": {
                            "type": "path",
                            "value": str(api.image_root / f"p{index_number}.png"),
                        },
                    }
                ]
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(index, range(12)))

    assert all(r.status_code == 200 for r in responses)
    assert api.get("/collections/c/stats").json()["points_count"] == 12


def test_searching_while_indexing_never_errors(api) -> None:
    api.post("/collections", json={"name": "c"})
    api.post(
        "/collections/c/index",
        json={
            "images": [
                {
                    "id": "seed",
                    "source": {"type": "path", "value": str(api.image_root / "p0.png")},
                }
            ]
        },
    )

    def work(index_number: int):
        if index_number % 2:
            return api.post(
                "/collections/c/index",
                json={
                    "images": [
                        {
                            "id": f"p{index_number}",
                            "source": {
                                "type": "path",
                                "value": str(api.image_root / f"p{index_number}.png"),
                            },
                        }
                    ]
                },
            )
        return api.post(
            "/collections/c/search",
            json={
                "source": {"type": "path", "value": str(api.image_root / "p0.png")},
                "top_k": 3,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(work, range(12)))

    assert all(r.status_code == 200 for r in responses), [
        r.status_code for r in responses
    ]


def test_concurrent_deletes_of_the_same_image_agree(api) -> None:
    api.post("/collections", json={"name": "c"})
    api.post(
        "/collections/c/index",
        json={
            "images": [
                {
                    "id": "target",
                    "source": {"type": "path", "value": str(api.image_root / "p1.png")},
                }
            ]
        },
    )

    with ThreadPoolExecutor(max_workers=6) as pool:
        statuses = [
            future.status_code
            for future in pool.map(
                lambda _: api.delete("/collections/c/images/target"), range(6)
            )
        ]

    assert statuses.count(200) >= 1
    assert all(status in {200, 404} for status in statuses)
    assert api.get("/collections/c/images/target").status_code == 404
