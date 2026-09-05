from fastapi.testclient import TestClient

from app.dependencies import get_indexing_service
from app.main import app
from app.schemas.index import IndexResponse


class FakeIndexingService:
    def __init__(self) -> None:
        self.calls = []

    def index_folder(self, **kwargs):
        self.calls.append(kwargs)
        return IndexResponse(
            collection_name=kwargs["collection_name"],
            indexed_count=2,
            failed_count=0,
            ids=["a", "nested__b"],
            errors=[],
        )


def test_folder_index_endpoint() -> None:
    service = FakeIndexingService()
    app.dependency_overrides[get_indexing_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/index/folder",
        json={
            "folder_path": "/data/images/products",
            "recursive": True,
            "metadata": {"category": "products"},
        },
    )

    app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 200
    assert body["indexed_count"] == 2
    assert body["failed_count"] == 0
    assert service.calls[0] == {
        "collection_name": "products",
        "folder_path": "/data/images/products",
        "recursive": True,
        "metadata": {"category": "products"},
    }
