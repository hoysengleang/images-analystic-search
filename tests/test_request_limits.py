"""Size and count limits that keep one request from exhausting the service."""

import io

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings
from app.dependencies import get_indexing_service, get_search_service
from app.main import create_app
from app.schemas.index import (
    MAX_IMAGES_PER_REQUEST,
    IndexImagesRequest,
    IndexResponse,
)
from app.schemas.search import MAX_QUERY_IMAGES, BatchSearchImagesRequest


class UnreachableService:
    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise AssertionError(f"{name} should not run for a rejected request")

        return fail


@pytest.fixture
def client() -> TestClient:
    app = create_app(Settings(max_image_size_mb=1, max_request_body_mb=2))
    app.dependency_overrides[get_indexing_service] = UnreachableService
    app.dependency_overrides[get_search_service] = UnreachableService
    return TestClient(app)


def upload_of(size_mb: float) -> dict:
    body = b"\xff" * int(size_mb * 1024 * 1024)
    return {"image": ("big.png", io.BytesIO(body), "image/png")}


def test_body_over_the_request_cap_is_rejected_before_parsing(
    client: TestClient,
) -> None:
    response = client.post(
        "/collections/products/index/upload",
        data={"id": "big"},
        files=upload_of(30),
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_BODY_TOO_LARGE"


def test_image_over_the_image_cap_is_rejected_while_reading(
    client: TestClient,
) -> None:
    response = client.post(
        "/collections/products/index/upload",
        data={"id": "big"},
        files=upload_of(1.5),
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_oversized_search_upload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/collections/products/search/upload",
        files=upload_of(1.5),
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


class AcceptingIndexingService:
    def index_upload(self, *, image_bytes, **kwargs):
        return IndexResponse(
            collection_name=kwargs["collection_name"],
            indexed_count=1,
            failed_count=0,
            ids=[kwargs["image_id"]],
            errors=[],
        )


def test_settings_passed_to_create_app_reach_the_routes() -> None:
    """The same upload must be judged by each app's own configured limits."""
    strict = create_app(Settings(max_image_size_mb=1, max_request_body_mb=2))
    generous = create_app(Settings(max_image_size_mb=10, max_request_body_mb=20))
    strict.dependency_overrides[get_indexing_service] = UnreachableService
    generous.dependency_overrides[get_indexing_service] = AcceptingIndexingService

    strict_response = TestClient(strict).post(
        "/collections/p/index/upload", data={"id": "x"}, files=upload_of(1.5)
    )
    generous_response = TestClient(generous).post(
        "/collections/p/index/upload", data={"id": "x"}, files=upload_of(1.5)
    )

    assert strict_response.status_code == 400
    assert strict_response.json()["error"]["code"] == "IMAGE_TOO_LARGE"
    assert generous_response.status_code == 200


def test_index_request_caps_the_number_of_images() -> None:
    too_many = [
        {
            "id": f"i{n}",
            "source": {"type": "url", "value": f"https://cdn.example.com/{n}.jpg"},
        }
        for n in range(MAX_IMAGES_PER_REQUEST + 1)
    ]

    with pytest.raises(ValidationError):
        IndexImagesRequest(images=too_many)

    assert len(IndexImagesRequest(images=too_many[:MAX_IMAGES_PER_REQUEST]).images) == (
        MAX_IMAGES_PER_REQUEST
    )


def test_batch_search_caps_the_number_of_query_images() -> None:
    too_many = [
        {"type": "url", "value": f"https://cdn.example.com/{n}.jpg"}
        for n in range(MAX_QUERY_IMAGES + 1)
    ]

    with pytest.raises(ValidationError):
        BatchSearchImagesRequest(sources=too_many)
