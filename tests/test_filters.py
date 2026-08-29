"""Filters must be applied or refused — never silently dropped."""

from pathlib import Path

import pytest
from PIL import Image
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.core.errors import BadRequestError
from app.embedding.base import EmbeddingProvider
from app.embedding.manager import EmbeddingManager
from app.embedding.registry import EmbeddingProviderRegistry
from app.schemas.collection import CollectionCreateRequest
from app.schemas.image import ImageSource
from app.schemas.index import IndexRequest
from app.schemas.search import SearchRequest
from app.services.collection_metadata_service import CollectionMetadataService
from app.services.collection_service import CollectionService
from app.services.image_loader import ImageLoader
from app.services.indexing_service import IndexingService
from app.services.search_service import SearchService
from app.services.vector_service import VectorService

PRODUCTS = [
    ("cheap_shoe", (255, 0, 0), {"category": "shoes", "price": 20, "in_stock": True}),
    ("mid_shoe", (200, 30, 0), {"category": "shoes", "price": 60, "in_stock": False}),
    ("cheap_bag", (0, 0, 255), {"category": "bags", "price": 30, "in_stock": True}),
    ("lux_bag", (0, 30, 200), {"category": "bags", "price": 400, "in_stock": True}),
]


class AverageColorProvider(EmbeddingProvider):
    provider_name = "average-color"

    def embed_image(self, image: Image.Image) -> list:
        pixels = list(image.convert("RGB").getdata())
        channels = [sum(pixel[c] for pixel in pixels) / len(pixels) for c in range(3)]
        norm = sum(v * v for v in channels) ** 0.5 or 1.0
        return [v / norm for v in channels]


@pytest.fixture
def catalogue(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name, color, _ in PRODUCTS:
        Image.new("RGB", (8, 8), color=color).save(image_root / f"{name}.png")

    settings = Settings(
        allowed_image_root=image_root,
        default_embedding_provider="average-color",
        default_model_name="average-color-v1",
        default_model_pretrained="none",
        default_vector_size=3,
    )
    registry = EmbeddingProviderRegistry()
    registry.register(AverageColorProvider)
    manager = EmbeddingManager(settings=settings, registry=registry)

    qdrant_client = QdrantClient(location=":memory:")
    metadata_service = CollectionMetadataService(tmp_path / "collections.json")
    vector_service = VectorService(qdrant_client=qdrant_client)

    CollectionService(
        settings=settings,
        qdrant_client=qdrant_client,
        metadata_service=metadata_service,
        embedding_manager=manager,
    ).create_collection(CollectionCreateRequest(name="products"))

    common = {
        "settings": settings,
        "metadata_service": metadata_service,
        "image_loader": ImageLoader(settings=settings),
        "embedding_manager": manager,
        "vector_service": vector_service,
    }
    IndexingService(**common).index(
        IndexRequest(
            collection_name="products",
            images=[
                {
                    "id": name,
                    "source": {"type": "path", "value": str(image_root / f"{name}.png")},
                    "metadata": metadata,
                }
                for name, _, metadata in PRODUCTS
            ],
        )
    )

    return SearchService(**common), image_root


def search(catalogue, filters):
    service, image_root = catalogue
    return service.search(
        SearchRequest(
            collection_name="products",
            source=ImageSource(type="path", value=str(image_root / "cheap_shoe.png")),
            top_k=10,
            filters=filters,
        )
    )


def test_scalar_filter_matches_exactly(catalogue) -> None:
    found = {result.id for result in search(catalogue, {"category": "bags"}).results}

    assert found == {"cheap_bag", "lux_bag"}


def test_boolean_filter_matches(catalogue) -> None:
    found = {result.id for result in search(catalogue, {"in_stock": False}).results}

    assert found == {"mid_shoe"}


def test_list_filter_matches_any_value(catalogue) -> None:
    found = {
        result.id for result in search(catalogue, {"category": ["bags", "shoes"]}).results
    }

    assert found == {"cheap_shoe", "mid_shoe", "cheap_bag", "lux_bag"}


def test_range_filter_narrows_by_number(catalogue) -> None:
    found = {
        result.id
        for result in search(catalogue, {"price": {"gte": 25, "lte": 100}}).results
    }

    assert found == {"mid_shoe", "cheap_bag"}


def test_combined_filters_are_all_applied(catalogue) -> None:
    found = {
        result.id
        for result in search(
            catalogue, {"category": "bags", "price": {"gt": 100}}
        ).results
    }

    assert found == {"lux_bag"}


@pytest.mark.parametrize(
    "filters",
    [
        # Keys Qdrant cannot address as a payload path. These must be a client
        # error, not a backend outage.
        {"a'; DROP TABLE products; --": "x"},
        {"weird key!": "x"},
        {"": "x"},
        {"trailing.": "x"},
        {"price": {"between": 10}},
        {"price": {"gte": "ten"}},
        {"tags": [1, "two"]},
        {"price": {}},
        {"nested": {"deep": {"gte": 1}}},
    ],
)
def test_filters_the_server_cannot_apply_are_refused(catalogue, filters) -> None:
    """A dropped filter would return unfiltered results that look correct."""
    with pytest.raises(BadRequestError) as exc_info:
        search(catalogue, filters)

    assert exc_info.value.code == "UNSUPPORTED_FILTER"
