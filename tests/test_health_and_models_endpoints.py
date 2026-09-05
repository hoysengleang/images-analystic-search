from fastapi.testclient import TestClient

from app.dependencies import get_embedding_manager, get_qdrant_client
from app.embedding.base import EmbeddingModelMetadata
from app.main import app


class FakeQdrantInfo:
    version = "1.12.0"


class FakeQdrantClient:
    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails

    def info(self):
        if self.fails:
            raise ConnectionError("connection refused")
        return FakeQdrantInfo()


class FakeEmbeddingManager:
    def get_default_model_metadata(self) -> EmbeddingModelMetadata:
        return EmbeddingModelMetadata(
            provider="openclip",
            model_name="ViT-B-32",
            model_pretrained="laion2b_s34b_b79k",
            vector_size=512,
        )

    def list_available_models(self):
        return [self.get_default_model_metadata()]


def test_health_does_not_touch_qdrant_by_default() -> None:
    app.dependency_overrides[get_qdrant_client] = lambda: FakeQdrantClient(fails=True)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["search_engine"] == "qdrant"
    assert "qdrant" not in body, "the probe must be opt-in"


def test_health_reports_qdrant_when_requested() -> None:
    app.dependency_overrides[get_qdrant_client] = lambda: FakeQdrantClient()
    client = TestClient(app)

    body = client.get("/health", params={"include_qdrant": "true"}).json()

    assert body["status"] == "ok"
    assert body["qdrant"]["status"] == "ok"
    assert body["qdrant"]["version"] == "1.12.0"


def test_health_reports_unavailable_qdrant_without_failing() -> None:
    app.dependency_overrides[get_qdrant_client] = lambda: FakeQdrantClient(fails=True)
    client = TestClient(app)

    response = client.get("/health", params={"include_qdrant": "true"})

    assert response.status_code == 200
    assert response.json()["qdrant"]["status"] == "unavailable"


def test_models_endpoint_lists_the_default_model() -> None:
    app.dependency_overrides[get_embedding_manager] = FakeEmbeddingManager
    client = TestClient(app)

    body = client.get("/models").json()

    assert body["default_model"]["model_name"] == "ViT-B-32"
    assert body["default_model"]["is_default"] is True
    assert body["models"][0]["vector_size"] == 512
