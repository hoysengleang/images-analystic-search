from fastapi.testclient import TestClient

from app.core.errors import ResourceNotFoundError
from app.dependencies import get_vector_service
from app.main import app
from app.schemas.image_record import ImageDeleteResponse, ImageRecordResponse


class FakeVectorService:
    def __init__(self) -> None:
        self.deleted = []

    def get_image(self, *, collection_name: str, image_id: str):
        if image_id == "missing":
            raise ResourceNotFoundError(
                message="Image not found: missing",
                code="IMAGE_NOT_FOUND",
                details={"collection_name": collection_name, "image_id": image_id},
            )
        return ImageRecordResponse(
            id=image_id,
            collection_name=collection_name,
            source_type="url",
            source_value="https://example.com/a.jpg",
            metadata={"name": "Blue Shoe"},
            embedding_provider="openclip",
            embedding_model="ViT-B-32",
            created_at="2026-05-10T00:00:00Z",
        )

    def delete_image_record(self, *, collection_name: str, image_id: str):
        self.get_image(collection_name=collection_name, image_id=image_id)
        self.deleted.append((collection_name, image_id))
        return ImageDeleteResponse(
            id=image_id,
            collection_name=collection_name,
            deleted=True,
        )


def test_get_image_payload() -> None:
    service = FakeVectorService()
    app.dependency_overrides[get_vector_service] = lambda: service
    client = TestClient(app)

    response = client.get("/collections/products/images/img_001")

    app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 200
    assert body["id"] == "img_001"
    assert body["collection_name"] == "products"
    assert body["source_type"] == "url"
    assert body["source_value"] == "https://example.com/a.jpg"
    assert body["metadata"] == {"name": "Blue Shoe"}
    assert body["embedding_provider"] == "openclip"
    assert body["embedding_model"] == "ViT-B-32"


def test_delete_image_vector() -> None:
    service = FakeVectorService()
    app.dependency_overrides[get_vector_service] = lambda: service
    client = TestClient(app)

    response = client.delete("/collections/products/images/img_001")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "id": "img_001",
        "collection_name": "products",
        "deleted": True,
    }
    assert service.deleted == [("products", "img_001")]


def test_get_missing_image_returns_clear_error() -> None:
    service = FakeVectorService()
    app.dependency_overrides[get_vector_service] = lambda: service
    client = TestClient(app)

    response = client.get("/collections/products/images/missing")

    app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 404
    assert body["error"]["code"] == "IMAGE_NOT_FOUND"
    assert body["error"]["details"] == {
        "collection_name": "products",
        "image_id": "missing",
    }
