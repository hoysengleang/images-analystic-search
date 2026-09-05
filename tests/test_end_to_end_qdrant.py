"""End-to-end run against a real Qdrant client in local (in-memory) mode.

The other tests use fakes for speed. This one exercises the actual
qdrant-client API — collection creation, upsert, query, scroll, delete — so a
mismatch with the real client surfaces here instead of in production.
"""

from pathlib import Path

import pytest
from PIL import Image
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.core.errors import ResourceNotFoundError
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

COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}


def normalized(channels) -> list:
    norm = sum(value * value for value in channels) ** 0.5 or 1.0
    return [value / norm for value in channels]


class AverageColorProvider(EmbeddingProvider):
    """Deterministic stand-in for OpenCLIP.

    Images encode to their normalized average color, and color words encode to
    the same axis — a toy version of the shared image/text vector space that
    makes text and hybrid search work.
    """

    provider_name = "average-color"
    supports_text = True

    def embed_image(self, image: Image.Image) -> list:
        pixels = list(image.convert("RGB").getdata())
        return normalized(
            [
                sum(pixel[channel] for pixel in pixels) / len(pixels)
                for channel in range(3)
            ]
        )

    def embed_text(self, text: str) -> list:
        words = text.lower()
        for name, color in COLORS.items():
            if name in words:
                return normalized(color)
        return normalized((1, 1, 1))


@pytest.fixture
def stack(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name, color in COLORS.items():
        Image.new("RGB", (8, 8), color=color).save(image_root / f"{name}.png")

    settings = Settings(
        allowed_image_root=image_root,
        default_embedding_provider="average-color",
        default_model_name="average-color-v1",
        default_model_pretrained="none",
        default_vector_size=3,
        embed_batch_size=2,
    )

    registry = EmbeddingProviderRegistry()
    registry.register(AverageColorProvider)
    embedding_manager = EmbeddingManager(settings=settings, registry=registry)

    qdrant_client = QdrantClient(location=":memory:")
    metadata_service = CollectionMetadataService(tmp_path / "collections.json")
    vector_service = VectorService(qdrant_client=qdrant_client)

    common = {
        "settings": settings,
        "metadata_service": metadata_service,
        "image_loader": ImageLoader(settings=settings),
        "embedding_manager": embedding_manager,
        "vector_service": vector_service,
    }

    return {
        "image_root": image_root,
        "collections": CollectionService(
            settings=settings,
            qdrant_client=qdrant_client,
            metadata_service=metadata_service,
            embedding_manager=embedding_manager,
        ),
        "indexing": IndexingService(**common),
        "search": SearchService(**common),
        "vectors": vector_service,
    }


def index_all_colors(stack) -> None:
    stack["collections"].create_collection(CollectionCreateRequest(name="swatches"))
    response = stack["indexing"].index(
        IndexRequest(
            collection_name="swatches",
            images=[
                {
                    "id": name,
                    "source": {
                        "type": "path",
                        "value": str(stack["image_root"] / f"{name}.png"),
                    },
                    "metadata": {"color": name},
                }
                for name in COLORS
            ],
        )
    )
    assert response.failed_count == 0, response.errors
    assert response.indexed_count == 3


def test_index_then_search_finds_the_matching_image(stack) -> None:
    index_all_colors(stack)

    response = stack["search"].search(
        SearchRequest(
            collection_name="swatches",
            source={"type": "path", "value": str(stack["image_root"] / "red.png")},
            top_k=3,
        )
    )

    assert response.results[0].id == "red"
    assert response.results[0].score == pytest.approx(1.0, abs=1e-5)
    assert response.results[0].metadata == {"color": "red"}


def test_metadata_filter_narrows_the_search(stack) -> None:
    index_all_colors(stack)

    response = stack["search"].search(
        SearchRequest(
            collection_name="swatches",
            source={"type": "path", "value": str(stack["image_root"] / "red.png")},
            top_k=3,
            filters={"color": "blue"},
        )
    )

    assert [result.id for result in response.results] == ["blue"]


def test_min_score_drops_distant_matches(stack) -> None:
    index_all_colors(stack)

    response = stack["search"].search(
        SearchRequest(
            collection_name="swatches",
            source={"type": "path", "value": str(stack["image_root"] / "red.png")},
            top_k=3,
            min_score=0.9,
        )
    )

    assert [result.id for result in response.results] == ["red"]


def test_records_can_be_read_listed_and_deleted(stack) -> None:
    index_all_colors(stack)

    record = stack["vectors"].get_image(collection_name="swatches", image_id="green")
    assert record.source_type == "path"
    assert record.embedding_model == "average-color-v1"

    listing = stack["vectors"].list_images(collection_name="swatches", limit=10)
    assert sorted(image.id for image in listing.images) == ["blue", "green", "red"]

    stack["vectors"].delete_image_record(collection_name="swatches", image_id="green")

    with pytest.raises(ResourceNotFoundError):
        stack["vectors"].get_image(collection_name="swatches", image_id="green")

    assert stack["collections"].get_collection_stats("swatches").points_count == 2


def test_reindexing_the_same_id_overwrites_it(stack) -> None:
    index_all_colors(stack)

    stack["indexing"].index(
        IndexRequest(
            collection_name="swatches",
            images=[
                {
                    "id": "red",
                    "source": {
                        "type": "path",
                        "value": str(stack["image_root"] / "blue.png"),
                    },
                    "metadata": {"color": "recolored"},
                }
            ],
        )
    )

    record = stack["vectors"].get_image(collection_name="swatches", image_id="red")
    assert record.metadata == {"color": "recolored"}
    assert stack["collections"].get_collection_stats("swatches").points_count == 3


def test_deleting_the_collection_removes_everything(stack) -> None:
    index_all_colors(stack)

    stack["collections"].delete_collection("swatches")

    assert stack["collections"].list_collections() == []
    with pytest.raises(ResourceNotFoundError):
        stack["collections"].get_collection_stats("swatches")


def test_text_query_finds_the_matching_image(stack) -> None:
    index_all_colors(stack)

    response = stack["search"].search_text(
        collection_name="swatches",
        query="a bright red swatch",
        top_k=3,
    )

    assert response.results[0].id == "red"
    assert response.results[0].score == pytest.approx(1.0, abs=1e-5)


def test_text_query_respects_metadata_filters(stack) -> None:
    index_all_colors(stack)

    response = stack["search"].search_text(
        collection_name="swatches",
        query="red",
        top_k=3,
        filters={"color": "green"},
    )

    assert [result.id for result in response.results] == ["green"]


@pytest.mark.parametrize(
    ("text_weight", "expected_top_result"),
    [(0.0, "red"), (1.0, "blue")],
)
def test_hybrid_weight_shifts_the_winner(
    stack,
    text_weight: float,
    expected_top_result: str,
) -> None:
    index_all_colors(stack)

    response = stack["search"].search_hybrid(
        collection_name="swatches",
        source=ImageSource(
            type="path",
            value=str(stack["image_root"] / "red.png"),
        ),
        query="blue",
        text_weight=text_weight,
        top_k=3,
    )

    assert response.results[0].id == expected_top_result
