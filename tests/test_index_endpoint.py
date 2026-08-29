from fastapi.testclient import TestClient

from app.dependencies import get_indexing_service
from app.main import app
from app.schemas.index import IndexResponse


class FakeIndexingService:
    def __init__(self) -> None:
        self.requests = []

    def index(self, request):
        self.requests.append(request)
        errors = []
        ids = []
        for item in request.images:
            if item.id == "broken":
                errors.append(
                    {
                        "id": item.id,
                        "code": "INVALID_IMAGE",
                        "message": "Invalid image file",
                        "details": {},
                    }
                )
            else:
                ids.append(item.id)

        return IndexResponse(
            collection_name=request.collection_name,
            indexed_count=len(ids),
            failed_count=len(errors),
            ids=ids,
            errors=errors,
        )


def test_index_url_image() -> None:
    service = FakeIndexingService()
    app.dependency_overrides[get_indexing_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/index",
        json={
            "images": [
                {
                    "id": "img_001",
                    "source": {
                        "type": "url",
                        "value": "https://example.com/a.jpg",
                    },
                    "metadata": {"name": "Blue Shoe"},
                }
            ]
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["indexed_count"] == 1
    assert response.json()["failed_count"] == 0
    assert service.requests[0].collection_name == "products"
    assert service.requests[0].images[0].source.type == "url"


def test_index_path_image() -> None:
    service = FakeIndexingService()
    app.dependency_overrides[get_indexing_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/index",
        json={
            "images": [
                {
                    "id": "img_001",
                    "source": {
                        "type": "path",
                        "value": "/data/images/a.jpg",
                    },
                }
            ]
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["indexed_count"] == 1
    assert service.requests[0].images[0].source.type == "path"


def test_index_base64_image() -> None:
    service = FakeIndexingService()
    app.dependency_overrides[get_indexing_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/index",
        json={
            "images": [
                {
                    "id": "img_001",
                    "source": {
                        "type": "base64",
                        "value": "abc123",
                    },
                }
            ]
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["indexed_count"] == 1
    assert service.requests[0].images[0].source.type == "base64"


def test_index_response_includes_failures() -> None:
    service = FakeIndexingService()
    app.dependency_overrides[get_indexing_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/index",
        json={
            "images": [
                {
                    "id": "ok",
                    "source": {
                        "type": "base64",
                        "value": "abc123",
                    },
                },
                {
                    "id": "broken",
                    "source": {
                        "type": "base64",
                        "value": "abc123",
                    },
                },
            ]
        },
    )

    app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 200
    assert body["indexed_count"] == 1
    assert body["failed_count"] == 1
    assert body["errors"][0]["id"] == "broken"
    assert body["errors"][0]["code"] == "INVALID_IMAGE"
