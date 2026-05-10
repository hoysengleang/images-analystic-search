from fastapi.testclient import TestClient

from app.api.routes.collections import get_search_service
from app.main import app
from app.schemas.search import SearchResponse, SearchResult


class FakeSearchService:
    def __init__(self) -> None:
        self.requests = []

    def search(self, request):
        self.requests.append(request)
        results = [
            SearchResult(
                id="img_001",
                score=0.92,
                source_type="url",
                source_value="https://example.com/a.jpg",
                metadata={"name": "Blue Shoe"},
                display_image_url="https://cdn.example.com/a.jpg",
            ),
            SearchResult(
                id="img_002",
                score=0.61,
                source_type="path",
                source_value="/data/images/b.jpg",
                metadata={"name": "Green Shoe"},
            ),
        ][: request.top_k]

        if request.min_score is not None:
            results = [
                result for result in results if result.score >= request.min_score
            ]

        return SearchResponse(
            collection_name=request.collection_name,
            top_k=request.top_k,
            results=results,
        )


def test_search_returns_similar_images() -> None:
    service = FakeSearchService()
    app.dependency_overrides[get_search_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/search",
        json={
            "source": {
                "type": "url",
                "value": "https://example.com/query.jpg",
            },
            "top_k": 10,
            "min_score": 0.7,
        },
    )

    app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 200
    assert body["collection_name"] == "products"
    assert body["top_k"] == 10
    assert body["results"] == [
        {
            "id": "img_001",
            "score": 0.92,
            "source_type": "url",
            "source_value": "https://example.com/a.jpg",
            "metadata": {"name": "Blue Shoe"},
            "display_image_url": "https://cdn.example.com/a.jpg",
        }
    ]
    assert service.requests[0].collection_name == "products"
    assert service.requests[0].source.type == "url"
    assert service.requests[0].min_score == 0.7


def test_search_top_k_limits_results() -> None:
    service = FakeSearchService()
    app.dependency_overrides[get_search_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/search",
        json={
            "source": {
                "type": "base64",
                "value": "abc123",
            },
            "top_k": 1,
        },
    )

    app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 200
    assert body["top_k"] == 1
    assert len(body["results"]) == 1
    assert service.requests[0].top_k == 1


def test_search_min_score_filters_low_scores() -> None:
    service = FakeSearchService()
    app.dependency_overrides[get_search_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/collections/products/search",
        json={
            "source": {
                "type": "path",
                "value": "/data/images/query.jpg",
            },
            "top_k": 10,
            "min_score": 0.95,
        },
    )

    app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 200
    assert body["results"] == []
    assert service.requests[0].source.type == "path"
    assert service.requests[0].min_score == 0.95
