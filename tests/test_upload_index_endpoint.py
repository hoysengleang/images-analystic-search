import json
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.dependencies import get_indexing_service
from app.embedding.base import EmbeddingProvider
from app.main import app
from app.schemas.collection import CollectionModelConfig
from app.services.collection_metadata_service import CollectionMetadataService
from app.services.indexing_service import IndexingService


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


def make_image_bytes() -> bytes:
    image = Image.new("RGB", (2, 2), color=(0, 0, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_indexing_service(tmp_path: Path):
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
    service = IndexingService(
        settings=Settings(),
        metadata_service=metadata_service,
        image_loader=FakeImageLoader(),
        embedding_manager=FakeEmbeddingManager(),
        vector_service=vector_service,
    )
    return service, vector_service


def test_upload_image_and_index_it(tmp_path: Path) -> None:
    service, vector_service = make_indexing_service(tmp_path)
    app.dependency_overrides[get_indexing_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/index/upload",
        data={
            "id": "img_001",
            "metadata": json.dumps({"name": "Blue Shoe"}),
        },
        files={
            "image": ("shoe.png", make_image_bytes(), "image/png"),
        },
    )

    app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 200
    assert body["indexed_count"] == 1
    assert body["failed_count"] == 0
    assert body["ids"] == ["img_001"]
    assert vector_service.upserts[0]["image_id"] == "img_001"
    assert vector_service.upserts[0]["source"].type == "upload"
    assert vector_service.upserts[0]["source"].value == "shoe.png"
    assert vector_service.upserts[0]["metadata"]["name"] == "Blue Shoe"
    assert vector_service.upserts[0]["metadata"]["original_filename"] == "shoe.png"
    assert vector_service.upserts[0]["metadata"]["content_type"] == "image/png"


def test_upload_invalid_file_is_rejected(tmp_path: Path) -> None:
    service, _ = make_indexing_service(tmp_path)
    app.dependency_overrides[get_indexing_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/index/upload",
        data={"id": "bad"},
        files={
            "image": ("bad.png", b"not an image", "image/png"),
        },
    )

    app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 400
    assert body["error"]["code"] == "INVALID_IMAGE"


def test_upload_invalid_metadata_is_rejected(tmp_path: Path) -> None:
    service, _ = make_indexing_service(tmp_path)
    app.dependency_overrides[get_indexing_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/index/upload",
        data={"id": "img_001", "metadata": "not-json"},
        files={
            "image": ("shoe.png", make_image_bytes(), "image/png"),
        },
    )

    app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 400
    assert body["error"]["code"] == "INVALID_UPLOAD_METADATA"
