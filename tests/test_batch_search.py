from pathlib import Path

from PIL import Image

from app.core.config import Settings
from app.embedding.base import EmbeddingProvider
from app.schemas.collection import CollectionModelConfig
from app.schemas.image import ImageSource
from app.schemas.search import SearchResult
from app.services.collection_metadata_service import CollectionMetadataService
from app.services.search_service import SearchService


class FakeImageLoader:
    def load_from_source(self, source):
        return Image.new("RGB", (2, 2))


class FakeEmbeddingProvider(EmbeddingProvider):
    provider_name = "openclip"

    def __init__(self) -> None:
        super().__init__(
            model_name="ViT-B-32",
            model_pretrained="laion2b_s34b_b79k",
            vector_size=2,
        )
        self.vectors = [[1.0, 0.0], [0.0, 1.0]]
        self.index = 0

    def embed_image(self, image: Image.Image) -> list[float]:
        vector = self.vectors[self.index]
        self.index += 1
        return vector


class FakeEmbeddingManager:
    def __init__(self) -> None:
        self.provider = FakeEmbeddingProvider()

    def get_provider(
        self, *, provider_name, model_name, model_pretrained, vector_size, **options
    ):
        return self.provider


class FakeVectorService:
    def __init__(self) -> None:
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return [
            SearchResult(
                id=f"result_{len(self.calls)}",
                score=0.9,
                source_type="url",
                source_value="https://example.com/a.jpg",
                metadata={"name": "Blue Shoe"},
            )
        ]


def make_service(tmp_path: Path):
    metadata_service = CollectionMetadataService(tmp_path / "collections.json")
    metadata_service.save(
        collection_name="products",
        model=CollectionModelConfig(
            provider="openclip",
            name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            vector_size=2,
            distance="cosine",
        ),
    )
    vector_service = FakeVectorService()
    service = SearchService(
        settings=Settings(default_top_k=10, max_top_k=100),
        metadata_service=metadata_service,
        image_loader=FakeImageLoader(),
        embedding_manager=FakeEmbeddingManager(),
        vector_service=vector_service,
    )
    return service, vector_service


def make_sources() -> list[ImageSource]:
    return [
        ImageSource(type="url", value="https://example.com/a.jpg"),
        ImageSource(type="path", value="/data/images/b.jpg"),
    ]


def test_average_mode_searches_once_with_normalized_average_vector(
    tmp_path: Path,
) -> None:
    service, vector_service = make_service(tmp_path)

    response = service.search_batch(
        collection_name="products",
        sources=make_sources(),
        mode="average",
        top_k=5,
        min_score=0.7,
    )

    assert response.mode == "average"
    assert response.results[0].id == "result_1"
    assert response.groups == []
    assert len(vector_service.calls) == 1
    assert vector_service.calls[0]["top_k"] == 5
    assert vector_service.calls[0]["min_score"] == 0.7
    assert vector_service.calls[0]["vector"] == [
        0.7071067811865475,
        0.7071067811865475,
    ]


def test_separate_mode_searches_each_source(tmp_path: Path) -> None:
    service, vector_service = make_service(tmp_path)

    response = service.search_batch(
        collection_name="products",
        sources=make_sources(),
        mode="separate",
        top_k=3,
    )

    assert response.mode == "separate"
    assert response.results == []
    assert len(response.groups) == 2
    assert response.groups[0].source_index == 0
    assert response.groups[0].source.type == "url"
    assert response.groups[0].results[0].id == "result_1"
    assert response.groups[1].source_index == 1
    assert response.groups[1].source.type == "path"
    assert response.groups[1].results[0].id == "result_2"
    assert len(vector_service.calls) == 2
    assert vector_service.calls[0]["vector"] == [1.0, 0.0]
    assert vector_service.calls[1]["vector"] == [0.0, 1.0]
