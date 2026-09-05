from fastapi.testclient import TestClient

from app.dependencies import get_search_service
from app.main import app
from app.schemas.search import BatchSearchResponse, SearchResult


class FakeSearchService:
    def __init__(self) -> None:
        self.calls = []

    def search_batch(self, **kwargs):
        self.calls.append(kwargs)
        return BatchSearchResponse(
            collection_name=kwargs["collection_name"],
            mode=kwargs["mode"],
            top_k=kwargs["top_k"],
            results=[
                SearchResult(
                    id="img_001",
                    score=0.93,
                    source_type="url",
                    source_value="https://example.com/a.jpg",
                    metadata={"name": "Blue Shoe"},
                )
            ],
            groups=[],
        )


def test_batch_search_endpoint() -> None:
    service = FakeSearchService()
    app.dependency_overrides[get_search_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/search/batch",
        json={
            "mode": "average",
            "sources": [
                {"type": "url", "value": "https://example.com/a.jpg"},
                {"type": "base64", "value": "abc123"},
            ],
            "top_k": 5,
            "min_score": 0.7,
        },
    )

    app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 200
    assert body["mode"] == "average"
    assert body["results"][0]["id"] == "img_001"
    assert service.calls[0]["collection_name"] == "products"
    assert service.calls[0]["mode"] == "average"
    assert service.calls[0]["top_k"] == 5
    assert service.calls[0]["min_score"] == 0.7
    assert service.calls[0]["sources"][0].type == "url"
    assert service.calls[0]["sources"][1].type == "base64"
