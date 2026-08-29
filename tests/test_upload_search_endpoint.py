from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.dependencies import get_search_service
from app.embedding.base import EmbeddingProvider
from app.main import app
from app.schemas.collection import CollectionModelConfig
from app.schemas.search import SearchResult
from app.services.collection_metadata_service import CollectionMetadataService
from app.services.search_service import SearchService


class FakeImageLoader:
    """Uploads and folder indexing read bytes directly, never via the loader."""

    def load_from_source(self, source):
        raise AssertionError("image loader should not be used on this path")


class FakeEmbeddingProvider(EmbeddingProvider):
    provider_name = "openclip"

    def __init__(self) -> None:
        super().__init__(
            model_name="ViT-B-32",
            model_pretrained="laion2b_s34b_b79k",
            vector_size=3,
        )

    def embed_image(self, image) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeEmbeddingManager:
    def get_provider(
        self, *, provider_name, model_name, model_pretrained, vector_size, **options
    ):
        return FakeEmbeddingProvider()


class FakeVectorService:
    def __init__(self) -> None:
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        results = [
            SearchResult(
                id="img_001",
                score=0.91,
                source_type="url",
                source_value="https://example.com/a.jpg",
                metadata={"name": "Blue Shoe"},
            ),
            SearchResult(
                id="img_002",
                score=0.52,
                source_type="path",
                source_value="/data/images/b.jpg",
                metadata={"name": "Green Shoe"},
            ),
        ][: kwargs["top_k"]]
        min_score = kwargs.get("min_score")
        if min_score is not None:
            results = [result for result in results if result.score >= min_score]
        return results


def make_image_bytes() -> bytes:
    image = Image.new("RGB", (2, 2), color=(255, 255, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_search_service(tmp_path: Path):
    metadata_service = CollectionMetadataService(tmp_path / "collections.json")
    metadata_service.save(
        collection_name="products",
        model=CollectionModelConfig(
            provider="openclip",
            name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            vector_size=3,
            distance="cosine",
        ),
    )
    vector_service = FakeVectorService()
    service = SearchService(
        settings=Settings(),
        metadata_service=metadata_service,
        image_loader=FakeImageLoader(),
        embedding_manager=FakeEmbeddingManager(),
        vector_service=vector_service,
    )
    return service, vector_service


def test_upload_query_image_and_search(tmp_path: Path) -> None:
    service, vector_service = make_search_service(tmp_path)
    app.dependency_overrides[get_search_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/search/upload",
        data={"top_k": "1"},
        files={"image": ("query.png", make_image_bytes(), "image/png")},
    )

    app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 200
    assert body["collection_name"] == "products"
    assert body["top_k"] == 1
    assert body["results"][0]["id"] == "img_001"
    assert body["results"][0]["metadata"] == {"name": "Blue Shoe"}
    assert vector_service.calls[0]["top_k"] == 1


def test_upload_query_min_score_filters_results(tmp_path: Path) -> None:
    service, vector_service = make_search_service(tmp_path)
    app.dependency_overrides[get_search_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/search/upload",
        data={"top_k": "10", "min_score": "0.95"},
        files={"image": ("query.png", make_image_bytes(), "image/png")},
    )

    app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 200
    assert body["results"] == []
    assert vector_service.calls[0]["min_score"] == 0.95


def test_upload_query_invalid_file_is_rejected(tmp_path: Path) -> None:
    service, _ = make_search_service(tmp_path)
    app.dependency_overrides[get_search_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/search/upload",
        data={"top_k": "10"},
        files={"image": ("query.png", b"not an image", "image/png")},
    )

    app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 400
    assert body["error"]["code"] == "INVALID_IMAGE"
