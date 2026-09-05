from pathlib import Path

import pytest
from PIL import Image

from app.core.config import Settings
from app.core.errors import InvalidImageError, ResourceNotFoundError
from app.embedding.base import EmbeddingProvider
from app.schemas.collection import CollectionModelConfig
from app.schemas.index import IndexRequest
from app.services.collection_metadata_service import CollectionMetadataService
from app.services.indexing_service import IndexingService


class FakeImageLoader:
    def load_from_source(self, source):
        if source.value == "broken":
            raise InvalidImageError()
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

    def get_provider(
        self, *, provider_name, model_name, model_pretrained, vector_size, **options
    ):
        self.calls.append(
            {
                "provider_name": provider_name,
                "model_name": model_name,
                "model_pretrained": model_pretrained,
                "vector_size": vector_size,
                **options,
            }
        )
        return self.provider


class FakeVectorService:
    def __init__(self) -> None:
        self.upserts = []

    def upsert_images(self, *, collection_name, records, embedding):
        for record in records:
            self.upserts.append(
                {
                    "collection_name": collection_name,
                    "image_id": record.image_id,
                    "vector": record.vector,
                    "source": record.source,
                    "metadata": record.metadata,
                    "embedding": embedding,
                }
            )
        return [record.image_id for record in records]


def make_service(tmp_path: Path):
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
    embedding_manager = FakeEmbeddingManager()
    vector_service = FakeVectorService()
    service = IndexingService(
        settings=Settings(),
        metadata_service=metadata_service,
        image_loader=FakeImageLoader(),
        embedding_manager=embedding_manager,
        vector_service=vector_service,
    )
    return service, embedding_manager, vector_service


def test_can_index_one_image(tmp_path: Path) -> None:
    service, embedding_manager, vector_service = make_service(tmp_path)
    request = IndexRequest.model_validate(
        {
            "collection_name": "products",
            "images": [
                {
                    "id": "product_001",
                    "source": {"type": "base64", "value": "ok"},
                    "metadata": {"category": "shoes"},
                }
            ],
        }
    )

    response = service.index(request)

    assert response.indexed_count == 1
    assert response.failed_count == 0
    assert response.ids == ["product_001"]
    assert vector_service.upserts[0]["collection_name"] == "products"
    assert vector_service.upserts[0]["metadata"] == {"category": "shoes"}
    assert embedding_manager.calls[0]["provider_name"] == "openclip"
    assert embedding_manager.calls[0]["vector_size"] == 3


def test_can_index_multiple_images(tmp_path: Path) -> None:
    service, _, vector_service = make_service(tmp_path)
    request = IndexRequest.model_validate(
        {
            "collection_name": "products",
            "images": [
                {"id": "a", "source": {"type": "base64", "value": "ok"}},
                {"id": "b", "source": {"type": "base64", "value": "ok"}},
            ],
        }
    )

    response = service.index(request)

    assert response.indexed_count == 2
    assert response.failed_count == 0
    assert response.ids == ["a", "b"]
    assert [upsert["image_id"] for upsert in vector_service.upserts] == ["a", "b"]


def test_failed_images_do_not_break_successful_images(tmp_path: Path) -> None:
    service, _, vector_service = make_service(tmp_path)
    request = IndexRequest.model_validate(
        {
            "collection_name": "products",
            "images": [
                {"id": "a", "source": {"type": "base64", "value": "ok"}},
                {"id": "bad", "source": {"type": "base64", "value": "broken"}},
                {"id": "b", "source": {"type": "base64", "value": "ok"}},
            ],
        }
    )

    response = service.index(request)

    assert response.indexed_count == 2
    assert response.failed_count == 1
    assert response.ids == ["a", "b"]
    assert response.errors[0]["id"] == "bad"
    assert response.errors[0]["code"] == "INVALID_IMAGE"
    assert [upsert["image_id"] for upsert in vector_service.upserts] == ["a", "b"]


def test_does_not_index_into_missing_collection(tmp_path: Path) -> None:
    metadata_service = CollectionMetadataService(tmp_path / "collections.json")
    service = IndexingService(
        settings=Settings(),
        metadata_service=metadata_service,
        image_loader=FakeImageLoader(),
        embedding_manager=FakeEmbeddingManager(),
        vector_service=FakeVectorService(),
    )
    request = IndexRequest.model_validate(
        {
            "collection_name": "missing",
            "images": [{"id": "a", "source": {"type": "base64", "value": "ok"}}],
        }
    )

    with pytest.raises(ResourceNotFoundError) as exc_info:
        service.index(request)

    assert exc_info.value.code == "COLLECTION_NOT_FOUND"
