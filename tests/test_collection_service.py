from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.errors import ConflictError, ResourceNotFoundError, ServiceUnavailableError
from app.embedding.base import EmbeddingModelMetadata
from app.schemas.collection import CollectionCreateRequest, CollectionModelConfig
from app.services.collection_metadata_service import CollectionMetadataService
from app.services.collection_service import CollectionService


class FakeCollectionInfo:
    def __init__(self, points_count: int) -> None:
        self.points_count = points_count


class FakeQdrantClient:
    def __init__(self, *, points_count: int = 0) -> None:
        self.collections: dict = {}
        self.points_count = points_count
        self.create_calls: list = []

    def collection_exists(self, *, collection_name):
        return collection_name in self.collections

    def create_collection(self, *, collection_name, vectors_config, metadata=None):
        self.create_calls.append(
            {
                "collection_name": collection_name,
                "size": vectors_config.size,
                "metadata": metadata,
            }
        )
        self.collections[collection_name] = vectors_config

    def delete_collection(self, *, collection_name):
        self.collections.pop(collection_name, None)

    def get_collection(self, *, collection_name):
        return FakeCollectionInfo(self.points_count)


class FakeEmbeddingManager:
    def get_default_model_metadata(self) -> EmbeddingModelMetadata:
        return EmbeddingModelMetadata(
            provider="openclip",
            model_name="ViT-B-32",
            model_pretrained="laion2b_s34b_b79k",
            vector_size=512,
        )


def make_service(tmp_path: Path, *, points_count: int = 0):
    qdrant_client = FakeQdrantClient(points_count=points_count)
    service = CollectionService(
        settings=Settings(),
        qdrant_client=qdrant_client,
        metadata_service=CollectionMetadataService(tmp_path / "collections.json"),
        embedding_manager=FakeEmbeddingManager(),
    )
    return service, qdrant_client


def test_create_collection_uses_the_default_model(tmp_path: Path) -> None:
    service, qdrant_client = make_service(tmp_path)

    response = service.create_collection(CollectionCreateRequest(name="products"))

    assert response.name == "products"
    assert response.model.name == "ViT-B-32"
    assert response.model.vector_size == 512
    assert qdrant_client.create_calls[0]["size"] == 512


def test_create_collection_accepts_an_explicit_model(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path)

    response = service.create_collection(
        CollectionCreateRequest(
            name="thumbnails",
            model=CollectionModelConfig(
                provider="openclip",
                name="ViT-L-14",
                pretrained="laion2b_s32b_b82k",
                vector_size=768,
            ),
        )
    )

    assert response.model.name == "ViT-L-14"
    assert response.model.vector_size == 768


def test_duplicate_collection_is_rejected(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path)
    service.create_collection(CollectionCreateRequest(name="products"))

    with pytest.raises(ConflictError) as exc_info:
        service.create_collection(CollectionCreateRequest(name="products"))

    assert exc_info.value.code == "COLLECTION_ALREADY_EXISTS"


def test_registry_and_qdrant_stay_in_sync_when_the_registry_fails(
    tmp_path: Path,
) -> None:
    service, qdrant_client = make_service(tmp_path)
    # A directory where the registry file belongs makes the write fail.
    (tmp_path / "collections.json").mkdir()

    with pytest.raises(OSError):
        service.create_collection(CollectionCreateRequest(name="products"))

    assert qdrant_client.collections == {}


def test_stats_report_the_live_point_count(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path, points_count=42)
    service.create_collection(CollectionCreateRequest(name="products"))

    stats = service.get_collection_stats("products")

    assert stats.points_count == 42
    assert stats.vector_size == 512
    assert stats.distance == "cosine"


def test_list_collections_returns_every_registered_collection(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path)
    service.create_collection(CollectionCreateRequest(name="products"))
    service.create_collection(CollectionCreateRequest(name="banners"))

    names = [collection.name for collection in service.list_collections()]

    assert names == ["banners", "products"]


def test_delete_collection_removes_it_everywhere(tmp_path: Path) -> None:
    service, qdrant_client = make_service(tmp_path)
    service.create_collection(CollectionCreateRequest(name="products"))

    service.delete_collection("products")

    assert qdrant_client.collections == {}
    assert service.list_collections() == []


def test_missing_collection_is_a_not_found_error(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path)

    with pytest.raises(ResourceNotFoundError) as exc_info:
        service.get_collection_stats("missing")

    assert exc_info.value.code == "COLLECTION_NOT_FOUND"


def test_qdrant_outage_surfaces_as_service_unavailable(tmp_path: Path) -> None:
    service, qdrant_client = make_service(tmp_path)

    def explode(**_kwargs):
        raise ConnectionError("qdrant is down")

    qdrant_client.collection_exists = explode

    with pytest.raises(ServiceUnavailableError) as exc_info:
        service.create_collection(CollectionCreateRequest(name="products"))

    assert exc_info.value.code == "QDRANT_COLLECTION_CHECK_FAILED"


def test_requested_distance_is_honoured(tmp_path: Path) -> None:
    service, qdrant_client = make_service(tmp_path)

    response = service.create_collection(
        CollectionCreateRequest(
            name="products",
            model=CollectionModelConfig(
                provider="openclip",
                name="ViT-B-32",
                pretrained="laion2b_s34b_b79k",
                vector_size=512,
                distance="dot",
            ),
        )
    )

    assert response.model.distance == "dot"
    assert qdrant_client.collections["products"].distance == "Dot"
    assert service.get_collection_stats("products").distance == "dot"


def test_default_distance_setting_is_used_when_none_is_given(tmp_path: Path) -> None:
    qdrant_client = FakeQdrantClient()
    service = CollectionService(
        settings=Settings(default_distance="euclid"),
        qdrant_client=qdrant_client,
        metadata_service=CollectionMetadataService(tmp_path / "collections.json"),
        embedding_manager=FakeEmbeddingManager(),
    )

    response = service.create_collection(CollectionCreateRequest(name="products"))

    assert response.model.distance == "euclid"
    assert qdrant_client.collections["products"].distance == "Euclid"


def test_concurrent_collection_creation_does_not_lose_registrations(
    tmp_path: Path,
) -> None:
    """A lost registry update would orphan a collection's vectors."""
    from concurrent.futures import ThreadPoolExecutor

    service, _ = make_service(tmp_path)

    def create(index: int) -> bool:
        try:
            service.create_collection(CollectionCreateRequest(name=f"c{index}"))
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=16) as pool:
        reported_success = sum(pool.map(create, range(40)))

    assert reported_success == 40
    assert len(service.list_collections()) == 40
