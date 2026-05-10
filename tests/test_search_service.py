from pathlib import Path

import pytest
from PIL import Image

from app.core.config import Settings
from app.core.errors import BadRequestError, ResourceNotFoundError
from app.embedding.base import EmbeddingProvider
from app.schemas.collection import CollectionModelConfig
from app.schemas.search import SearchRequest, SearchResult
from app.services.collection_metadata_service import CollectionMetadataService
from app.services.search_service import SearchService


class FakeImageLoader:
    def __init__(self) -> None:
        self.sources = []

    def load_from_source(self, source):
        self.sources.append(source)
        return Image.new("RGB", (2, 2))


class FakeEmbeddingProvider(EmbeddingProvider):
    provider_name = "openclip"

    def __init__(self) -> None:
        super().__init__(
            model_name="ViT-B-32",
            model_pretrained="laion2b_s34b_b79k",
            vector_size=3,
        )

    def embed_image(self, image: Image.Image) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeEmbeddingManager:
    def __init__(self) -> None:
        self.calls = []
        self.provider = FakeEmbeddingProvider()

    def get_provider(self, *, provider_name, model_name, model_pretrained, vector_size):
        self.calls.append(
            {
                "provider_name": provider_name,
                "model_name": model_name,
                "model_pretrained": model_pretrained,
                "vector_size": vector_size,
            }
        )
        return self.provider


class FakeVectorService:
    def __init__(self) -> None:
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        results = [
            SearchResult(
                id="img_001",
                score=0.98,
                source_type="url",
                source_value="https://example.com/a.jpg",
                metadata={"name": "Blue Shoe"},
                display_image_url="https://cdn.example.com/a.jpg",
            )
        ]
        min_score = kwargs.get("min_score")
        if min_score is not None:
            results = [result for result in results if result.score >= min_score]
        return results


def make_service(tmp_path: Path, *, max_top_k: int = 100):
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
    image_loader = FakeImageLoader()
    embedding_manager = FakeEmbeddingManager()
    vector_service = FakeVectorService()
    service = SearchService(
        settings=Settings(default_top_k=min(3, max_top_k), max_top_k=max_top_k),
        metadata_service=metadata_service,
        image_loader=image_loader,
        embedding_manager=embedding_manager,
        vector_service=vector_service,
    )
    return service, image_loader, embedding_manager, vector_service


@pytest.mark.parametrize(
    ("source_type", "source_value"),
    [
        ("url", "https://example.com/query.jpg"),
        ("path", "/data/images/query.jpg"),
        ("base64", "abc123"),
    ],
)
def test_can_search_by_supported_source_types(
    tmp_path: Path,
    source_type: str,
    source_value: str,
) -> None:
    service, image_loader, embedding_manager, vector_service = make_service(tmp_path)
    request = SearchRequest.model_validate(
        {
            "collection_name": "products",
            "source": {"type": source_type, "value": source_value},
            "top_k": 5,
        }
    )

    response = service.search(request)

    assert response.collection_name == "products"
    assert response.top_k == 5
    assert response.results[0].id == "img_001"
    assert response.results[0].score == 0.98
    assert response.results[0].source_type == "url"
    assert response.results[0].source_value == "https://example.com/a.jpg"
    assert response.results[0].metadata == {"name": "Blue Shoe"}
    assert image_loader.sources[0].type == source_type
    assert embedding_manager.calls[0]["provider_name"] == "openclip"
    assert vector_service.calls[0]["top_k"] == 5


def test_respects_min_score(tmp_path: Path) -> None:
    service, _, _, vector_service = make_service(tmp_path)
    request = SearchRequest.model_validate(
        {
            "collection_name": "products",
            "source": {"type": "base64", "value": "abc123"},
            "top_k": 5,
            "min_score": 0.99,
        }
    )

    response = service.search(request)

    assert response.results == []
    assert vector_service.calls[0]["min_score"] == 0.99


def test_does_not_search_missing_collection(tmp_path: Path) -> None:
    service = SearchService(
        settings=Settings(),
        metadata_service=CollectionMetadataService(tmp_path / "collections.json"),
        image_loader=FakeImageLoader(),
        embedding_manager=FakeEmbeddingManager(),
        vector_service=FakeVectorService(),
    )
    request = SearchRequest.model_validate(
        {
            "collection_name": "missing",
            "source": {"type": "base64", "value": "abc123"},
        }
    )

    with pytest.raises(ResourceNotFoundError) as exc_info:
        service.search(request)

    assert exc_info.value.code == "COLLECTION_NOT_FOUND"


def test_rejects_top_k_above_configured_max(tmp_path: Path) -> None:
    service, _, _, _ = make_service(tmp_path, max_top_k=3)
    request = SearchRequest.model_validate(
        {
            "collection_name": "products",
            "source": {"type": "base64", "value": "abc123"},
            "top_k": 5,
        }
    )

    with pytest.raises(BadRequestError) as exc_info:
        service.search(request)

    assert exc_info.value.code == "TOP_K_TOO_LARGE"
