from fastapi.testclient import TestClient

from app.dependencies import get_vector_service
from app.main import app
from app.services.vector_service import VectorService


class FakePoint:
    def __init__(self, image_id: str) -> None:
        self.id = f"point-{image_id}"
        self.payload = {
            "image_id": image_id,
            "source_type": "url",
            "source_value": f"https://cdn.example.com/{image_id}.jpg",
            "metadata": {"name": image_id},
            "embedding_provider": "openclip",
            "embedding_model": "ViT-B-32",
            "created_at": "2026-01-01T00:00:00+00:00",
        }


class FakeQdrantClient:
    def __init__(self, *, next_offset=None) -> None:
        self.next_offset = next_offset
        self.scroll_calls: list = []

    def scroll(self, *, collection_name, limit, offset, with_payload, with_vectors):
        self.scroll_calls.append({"limit": limit, "offset": offset})
        return [FakePoint("img_001"), FakePoint("img_002")], self.next_offset


def test_list_images_returns_stored_records() -> None:
    client_stub = FakeQdrantClient()
    service = VectorService(qdrant_client=client_stub)

    response = service.list_images(collection_name="products", limit=50)

    assert [image.id for image in response.images] == ["img_001", "img_002"]
    assert response.images[0].metadata == {"name": "img_001"}
    assert response.next_cursor is None


def test_list_images_reports_the_next_cursor() -> None:
    service = VectorService(qdrant_client=FakeQdrantClient(next_offset="point-img_002"))

    response = service.list_images(collection_name="products", limit=2)

    assert response.next_cursor == "point-img_002"


def test_list_images_endpoint_passes_pagination_through() -> None:
    client_stub = FakeQdrantClient(next_offset="point-img_002")
    app.dependency_overrides[get_vector_service] = lambda: VectorService(
        qdrant_client=client_stub
    )
    client = TestClient(app)

    body = client.get(
        "/collections/products/images",
        params={"limit": 2, "cursor": "point-img_000"},
    ).json()

    assert client_stub.scroll_calls == [{"limit": 2, "offset": "point-img_000"}]
    assert body["collection_name"] == "products"
    assert len(body["images"]) == 2
    assert body["next_cursor"] == "point-img_002"
