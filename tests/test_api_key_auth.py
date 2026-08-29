from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.dependencies import get_collection_service
from app.main import app


class FakeCollectionService:
    def list_collections(self):
        return []


def override_settings(**overrides) -> None:
    settings = Settings(**overrides)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_collection_service] = FakeCollectionService


def test_requests_pass_through_when_no_api_key_is_configured() -> None:
    override_settings()
    client = TestClient(app)

    assert client.get("/collections").status_code == 200


def test_request_without_key_is_rejected_when_a_key_is_configured() -> None:
    override_settings(api_key="s3cret")
    client = TestClient(app)

    response = client.get("/collections")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_request_with_wrong_key_is_rejected() -> None:
    override_settings(api_key="s3cret")
    client = TestClient(app)

    response = client.get("/collections", headers={"X-API-Key": "wrong"})

    assert response.status_code == 401


def test_request_with_correct_key_is_accepted() -> None:
    override_settings(api_key="s3cret")
    client = TestClient(app)

    response = client.get("/collections", headers={"X-API-Key": "s3cret"})

    assert response.status_code == 200


def test_health_stays_public_when_a_key_is_configured() -> None:
    override_settings(api_key="s3cret")
    client = TestClient(app)

    assert client.get("/health").status_code == 200
