from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app.core.config import Settings
from app.core.errors import BadRequestError
from app.embedding.base import EmbeddingProvider
from app.schemas.collection import CollectionModelConfig
from app.services.collection_metadata_service import CollectionMetadataService
from app.services.indexing_service import IndexingService


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
    def get_provider(self, *, provider_name, model_name, model_pretrained, vector_size):
        return FakeEmbeddingProvider()


class FakeVectorService:
    def __init__(self) -> None:
        self.upserts = []

    def upsert_image(self, **kwargs):
        self.upserts.append(kwargs)
        return kwargs["image_id"]


def make_image_bytes() -> bytes:
    image = Image.new("RGB", (2, 2), color=(255, 0, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_service(tmp_path: Path, allowed_root: Path):
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
        settings=Settings(allowed_image_root=allowed_root),
        metadata_service=metadata_service,
        embedding_manager=FakeEmbeddingManager(),
        vector_service=vector_service,
    )
    return service, vector_service


def test_indexes_images_in_folder(tmp_path: Path) -> None:
    allowed_root = tmp_path / "images"
    folder = allowed_root / "products"
    folder.mkdir(parents=True)
    (folder / "a.png").write_bytes(make_image_bytes())
    (folder / "notes.txt").write_text("ignore me", encoding="utf-8")
    nested = folder / "nested"
    nested.mkdir()
    (nested / "b.png").write_bytes(make_image_bytes())
    service, vector_service = make_service(tmp_path, allowed_root)

    response = service.index_folder(
        collection_name="products",
        folder_path=str(folder),
        recursive=False,
        metadata={"category": "products"},
    )

    assert response.indexed_count == 1
    assert response.failed_count == 0
    assert response.ids == ["a"]
    assert vector_service.upserts[0]["source"].type == "folder"
    assert vector_service.upserts[0]["metadata"]["category"] == "products"
    assert vector_service.upserts[0]["metadata"]["relative_path"] == "a.png"


def test_recursive_folder_indexing(tmp_path: Path) -> None:
    allowed_root = tmp_path / "images"
    folder = allowed_root / "products"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    (folder / "a.png").write_bytes(make_image_bytes())
    (nested / "b.png").write_bytes(make_image_bytes())
    service, vector_service = make_service(tmp_path, allowed_root)

    response = service.index_folder(
        collection_name="products",
        folder_path=str(folder),
        recursive=True,
        metadata={},
    )

    assert response.indexed_count == 2
    assert response.failed_count == 0
    assert response.ids == ["a", "nested__b"]
    assert [upsert["image_id"] for upsert in vector_service.upserts] == [
        "a",
        "nested__b",
    ]


def test_unsafe_folder_path_is_blocked(tmp_path: Path) -> None:
    allowed_root = tmp_path / "images"
    allowed_root.mkdir()
    service, _ = make_service(tmp_path, allowed_root)

    with pytest.raises(BadRequestError) as exc_info:
        service.index_folder(
            collection_name="products",
            folder_path="../../secret",
            recursive=True,
            metadata={},
        )

    assert exc_info.value.code == "IMAGE_PATH_NOT_ALLOWED"


def test_broken_image_counts_as_failure(tmp_path: Path) -> None:
    allowed_root = tmp_path / "images"
    folder = allowed_root / "products"
    folder.mkdir(parents=True)
    (folder / "a.png").write_bytes(make_image_bytes())
    (folder / "broken.png").write_bytes(b"not an image")
    service, vector_service = make_service(tmp_path, allowed_root)

    response = service.index_folder(
        collection_name="products",
        folder_path=str(folder),
        recursive=True,
        metadata={},
    )

    assert response.indexed_count == 1
    assert response.failed_count == 1
    assert response.ids == ["a"]
    assert response.errors[0]["id"] == "broken"
    assert response.errors[0]["code"] == "INVALID_IMAGE"
    assert [upsert["image_id"] for upsert in vector_service.upserts] == ["a"]
