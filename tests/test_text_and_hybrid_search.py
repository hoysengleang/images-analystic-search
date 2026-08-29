from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.core.errors import BadRequestError
from app.dependencies import get_search_service
from app.embedding.base import EmbeddingProvider
from app.main import app
from app.schemas.collection import CollectionModelConfig
from app.schemas.image import ImageSource
from app.schemas.search import SearchResult
from app.services.collection_metadata_service import CollectionMetadataService
from app.services.search_service import SearchService

IMAGE_VECTOR = [1.0, 0.0, 0.0]
TEXT_VECTOR = [0.0, 1.0, 0.0]


class FakeImageLoader:
    def load_from_source(self, source):
        return Image.new("RGB", (2, 2))


class TextCapableProvider(EmbeddingProvider):
    provider_name = "fake-multimodal"
    supports_text = True

    def __init__(self) -> None:
        super().__init__(
            model_name="fake",
            model_pretrained="none",
            vector_size=3,
        )

    def embed_image(self, image) -> list:
        return list(IMAGE_VECTOR)

    def embed_text(self, text: str) -> list:
        return list(TEXT_VECTOR)


class ImageOnlyProvider(EmbeddingProvider):
    provider_name = "fake-image-only"

    def __init__(self) -> None:
        super().__init__(model_name="fake", model_pretrained="none", vector_size=3)

    def embed_image(self, image) -> list:
        return list(IMAGE_VECTOR)


class FakeEmbeddingManager:
    def __init__(self, provider) -> None:
        self.provider = provider

    def get_provider(self, **_kwargs):
        return self.provider


class RecordingVectorService:
    def __init__(self) -> None:
        self.calls: list = []

    def search(self, *, collection_name, vector, top_k, filters=None, min_score=None):
        self.calls.append(
            {
                "collection_name": collection_name,
                "vector": vector,
                "top_k": top_k,
                "filters": filters,
                "min_score": min_score,
            }
        )
        return [SearchResult(id="img_001", score=0.9)]


def make_service(tmp_path: Path, provider):
    metadata_service = CollectionMetadataService(tmp_path / "collections.json")
    metadata_service.save(
        collection_name="products",
        model=CollectionModelConfig(
            provider=provider.provider_name,
            name="fake",
            pretrained="none",
            vector_size=3,
            distance="cosine",
        ),
    )
    vector_service = RecordingVectorService()
    service = SearchService(
        settings=Settings(default_top_k=10, max_top_k=100),
        metadata_service=metadata_service,
        image_loader=FakeImageLoader(),
        embedding_manager=FakeEmbeddingManager(provider),
        vector_service=vector_service,
    )
    return service, vector_service


def test_text_search_queries_with_the_text_vector(tmp_path: Path) -> None:
    service, vector_service = make_service(tmp_path, TextCapableProvider())

    response = service.search_text(
        collection_name="products",
        query="red running shoe",
        top_k=5,
        min_score=0.2,
        filters={"category": "shoes"},
    )

    assert response.results[0].id == "img_001"
    assert vector_service.calls[0]["vector"] == TEXT_VECTOR
    assert vector_service.calls[0]["top_k"] == 5
    assert vector_service.calls[0]["min_score"] == 0.2
    assert vector_service.calls[0]["filters"] == {"category": "shoes"}


def test_text_search_falls_back_to_the_configured_default_top_k(tmp_path: Path) -> None:
    service, vector_service = make_service(tmp_path, TextCapableProvider())

    response = service.search_text(collection_name="products", query="shoe", top_k=None)

    assert response.top_k == 10
    assert vector_service.calls[0]["top_k"] == 10


def test_text_search_is_refused_by_an_image_only_provider(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path, ImageOnlyProvider())

    with pytest.raises(BadRequestError) as exc_info:
        service.search_text(collection_name="products", query="shoe", top_k=5)

    assert exc_info.value.code == "TEXT_SEARCH_NOT_SUPPORTED"


@pytest.mark.parametrize(
    ("text_weight", "expected_vector"),
    [
        (0.0, IMAGE_VECTOR),
        (1.0, TEXT_VECTOR),
        (0.5, [0.7071067811865475, 0.7071067811865475, 0.0]),
    ],
)
def test_hybrid_weight_moves_the_query_between_image_and_text(
    tmp_path: Path,
    text_weight: float,
    expected_vector: list,
) -> None:
    service, vector_service = make_service(tmp_path, TextCapableProvider())

    service.search_hybrid(
        collection_name="products",
        source=ImageSource(type="url", value="https://cdn.example.com/a.jpg"),
        query="but in blue",
        text_weight=text_weight,
        top_k=5,
    )

    assert vector_service.calls[0]["vector"] == pytest.approx(expected_vector)


def test_hybrid_query_vector_stays_normalized(tmp_path: Path) -> None:
    service, vector_service = make_service(tmp_path, TextCapableProvider())

    service.search_hybrid(
        collection_name="products",
        source=ImageSource(type="url", value="https://cdn.example.com/a.jpg"),
        query="but in blue",
        text_weight=0.3,
        top_k=5,
    )

    vector = vector_service.calls[0]["vector"]
    assert sum(value * value for value in vector) == pytest.approx(1.0)


class FakeSearchService:
    def __init__(self) -> None:
        self.calls: list = []

    def search_text(self, **kwargs):
        self.calls.append(("text", kwargs))
        from app.schemas.search import SearchResponse

        return SearchResponse(
            collection_name=kwargs["collection_name"],
            top_k=kwargs["top_k"] or 10,
            results=[SearchResult(id="img_001", score=0.31)],
        )

    def search_hybrid(self, **kwargs):
        self.calls.append(("hybrid", kwargs))
        from app.schemas.search import SearchResponse

        return SearchResponse(
            collection_name=kwargs["collection_name"],
            top_k=kwargs["top_k"] or 10,
            results=[SearchResult(id="img_002", score=0.77)],
        )


def test_text_search_endpoint() -> None:
    service = FakeSearchService()
    app.dependency_overrides[get_search_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/search/text",
        json={"query": "  red running shoe  ", "top_k": 5, "min_score": 0.2},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "img_001"
    assert service.calls[0][1]["query"] == "red running shoe"
    assert service.calls[0][1]["top_k"] == 5


def test_hybrid_search_endpoint() -> None:
    service = FakeSearchService()
    app.dependency_overrides[get_search_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/search/hybrid",
        json={
            "source": {"type": "url", "value": "https://cdn.example.com/a.jpg"},
            "query": "but in blue",
            "text_weight": 0.4,
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "img_002"
    assert service.calls[0][1]["text_weight"] == 0.4


def test_empty_text_query_is_rejected() -> None:
    app.dependency_overrides[get_search_service] = FakeSearchService
    client = TestClient(app)

    response = client.post("/collections/products/search/text", json={"query": "   "})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_text_weight_outside_zero_to_one_is_rejected() -> None:
    app.dependency_overrides[get_search_service] = FakeSearchService
    client = TestClient(app)

    response = client.post(
        "/collections/products/search/hybrid",
        json={
            "source": {"type": "url", "value": "https://cdn.example.com/a.jpg"},
            "query": "blue",
            "text_weight": 1.5,
        },
    )

    assert response.status_code == 422
