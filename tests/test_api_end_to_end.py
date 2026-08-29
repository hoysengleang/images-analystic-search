"""Every endpoint, driven through the real service graph.

Nothing is faked below the HTTP layer except the embedding model: a real
Qdrant (in local mode), the real services, the real routes, the real
middleware. The stub encoder is deterministic so results are assertable, but
it goes through exactly the same code path OpenCLIP does.
"""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.dependencies import (
    get_collection_service,
    get_embedding_manager,
    get_indexing_service,
    get_qdrant_client,
    get_search_service,
    get_vector_service,
)
from app.embedding.base import EmbeddingProvider
from app.embedding.manager import EmbeddingManager
from app.embedding.registry import EmbeddingProviderRegistry
from app.main import create_app
from app.services.collection_metadata_service import CollectionMetadataService
from app.services.collection_service import CollectionService
from app.services.image_loader import ImageLoader
from app.services.indexing_service import IndexingService
from app.services.search_service import SearchService
from app.services.vector_service import VectorService

COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
}


def normalized(channels) -> list:
    norm = sum(value * value for value in channels) ** 0.5 or 1.0
    return [value / norm for value in channels]


class ColorProvider(EmbeddingProvider):
    """Deterministic stand-in for OpenCLIP with a matching text encoder."""

    provider_name = "color"
    supports_text = True

    def embed_image(self, image) -> list:
        pixels = list(image.convert("RGB").getdata())
        return normalized([sum(p[c] for p in pixels) / len(pixels) for c in range(3)])

    def embed_text(self, text: str) -> list:
        words = text.lower()
        for name, color in COLORS.items():
            if name in words:
                return normalized(color)
        return normalized((1, 1, 1))


def png_bytes(color, size=(16, 16)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def api(tmp_path):
    """The real app, wired to a real in-process Qdrant."""
    image_root = tmp_path / "images"
    (image_root / "catalogue").mkdir(parents=True)
    for name, color in COLORS.items():
        (image_root / f"{name}.png").write_bytes(png_bytes(color))
        (image_root / "catalogue" / f"{name}.png").write_bytes(png_bytes(color))
    (image_root / "catalogue" / "notes.txt").write_text("not an image")

    settings = Settings(
        allowed_image_root=image_root,
        default_embedding_provider="color",
        default_model_name="color-v1",
        default_model_pretrained="none",
        default_vector_size=3,
        embed_batch_size=2,
        default_top_k=5,
        max_top_k=50,
        max_image_size_mb=2,
        max_request_body_mb=4,
    )

    registry = EmbeddingProviderRegistry()
    registry.register(ColorProvider)
    manager = EmbeddingManager(settings=settings, registry=registry)
    qdrant_client = QdrantClient(location=":memory:")
    metadata_service = CollectionMetadataService(tmp_path / "collections.json")
    vector_service = VectorService(qdrant_client=qdrant_client)
    image_loader = ImageLoader(settings=settings)

    services = {
        "settings": settings,
        "metadata_service": metadata_service,
        "image_loader": image_loader,
        "embedding_manager": manager,
        "vector_service": vector_service,
    }

    app = create_app(settings)
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant_client
    app.dependency_overrides[get_embedding_manager] = lambda: manager
    app.dependency_overrides[get_vector_service] = lambda: vector_service
    app.dependency_overrides[get_collection_service] = lambda: CollectionService(
        settings=settings,
        qdrant_client=qdrant_client,
        metadata_service=metadata_service,
        embedding_manager=manager,
    )
    app.dependency_overrides[get_indexing_service] = lambda: IndexingService(**services)
    app.dependency_overrides[get_search_service] = lambda: SearchService(**services)

    client = TestClient(app)
    client.image_root = image_root
    return client


def seed(api) -> None:
    """Create a collection and index one image per colour, by local path."""
    assert api.post("/collections", json={"name": "swatches"}).status_code == 201
    response = api.post(
        "/collections/swatches/index",
        json={
            "images": [
                {
                    "id": name,
                    "source": {
                        "type": "path",
                        "value": str(api.image_root / f"{name}.png"),
                    },
                    "metadata": {"color": name, "price": 10 * index, "tags": [name]},
                    "display_image_url": f"https://cdn.example.com/{name}.png",
                }
                for index, name in enumerate(COLORS)
            ]
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["failed_count"] == 0, response.json()["errors"]


# --------------------------------------------------------------------------
# Service level
# --------------------------------------------------------------------------


def test_health(api) -> None:
    body = api.get("/health").json()
    assert body["status"] == "ok"
    assert body["app_name"] == "OpenVisionSearch"
    assert body["search_engine"] == "qdrant"
    assert "qdrant" not in body, "the probe must be opt-in"


def test_health_with_qdrant_probe(api) -> None:
    body = api.get("/health", params={"include_qdrant": "true"}).json()
    assert body["qdrant"]["status"] in {"ok", "unavailable"}


def test_models_endpoint(api) -> None:
    body = api.get("/models").json()
    assert body["default_model"]["model_name"] == "color-v1"
    assert body["default_model"]["supports_text"] is True


def test_openapi_and_docs_render(api) -> None:
    assert api.get("/openapi.json").status_code == 200
    assert api.get("/docs").status_code == 200
    assert api.get("/redoc").status_code == 200


# --------------------------------------------------------------------------
# Collections
# --------------------------------------------------------------------------


def test_collection_lifecycle(api) -> None:
    created = api.post("/collections", json={"name": "swatches"})
    assert created.status_code == 201
    assert created.json()["model"]["vector_size"] == 3

    assert [c["name"] for c in api.get("/collections").json()["collections"]] == [
        "swatches"
    ]

    stats = api.get("/collections/swatches/stats").json()
    assert stats["points_count"] == 0
    assert stats["distance"] == "cosine"

    assert api.delete("/collections/swatches").status_code == 200
    assert api.get("/collections").json()["collections"] == []


def test_duplicate_collection_is_a_conflict(api) -> None:
    api.post("/collections", json={"name": "swatches"})
    response = api.post("/collections", json={"name": "swatches"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "COLLECTION_ALREADY_EXISTS"


def test_stats_track_indexed_points(api) -> None:
    seed(api)
    assert api.get("/collections/swatches/stats").json()["points_count"] == len(COLORS)


@pytest.mark.parametrize(
    "path",
    [
        "/collections/missing/stats",
        "/collections/missing/images/x",
    ],
)
def test_missing_collection_is_reported(api, path) -> None:
    response = api.get(path)
    assert response.status_code in {404, 503}
    assert "error" in response.json()


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------


def test_index_by_path_upload_and_base64(api, tmp_path) -> None:
    api.post("/collections", json={"name": "swatches"})

    by_path = api.post(
        "/collections/swatches/index",
        json={
            "images": [
                {
                    "id": "red",
                    "source": {
                        "type": "path",
                        "value": str(api.image_root / "red.png"),
                    },
                }
            ]
        },
    ).json()
    assert by_path["indexed_count"] == 1

    uploaded = api.post(
        "/collections/swatches/index/upload",
        data={"id": "green", "metadata": '{"color":"green"}'},
        files={"image": ("green.png", BytesIO(png_bytes(COLORS["green"])), "image/png")},
    ).json()
    assert uploaded["indexed_count"] == 1

    import base64

    encoded = base64.b64encode(png_bytes(COLORS["blue"])).decode("ascii")
    by_base64 = api.post(
        "/collections/swatches/index",
        json={"images": [{"id": "blue", "source": {"type": "base64", "value": encoded}}]},
    ).json()
    assert by_base64["indexed_count"] == 1

    assert api.get("/collections/swatches/stats").json()["points_count"] == 3


def test_index_folder_skips_non_images(api) -> None:
    api.post("/collections", json={"name": "swatches"})

    response = api.post(
        "/collections/swatches/index/folder",
        json={"folder_path": str(api.image_root / "catalogue"), "recursive": True},
    ).json()

    assert response["indexed_count"] == len(COLORS)
    assert response["failed_count"] == 0


def test_partial_failure_reports_per_image(api) -> None:
    api.post("/collections", json={"name": "swatches"})

    response = api.post(
        "/collections/swatches/index",
        json={
            "images": [
                {
                    "id": "good",
                    "source": {
                        "type": "path",
                        "value": str(api.image_root / "red.png"),
                    },
                },
                {"id": "bad", "source": {"type": "base64", "value": "bm90YW5pbWFnZQ=="}},
            ]
        },
    ).json()

    assert response["indexed_count"] == 1
    assert response["failed_count"] == 1
    assert response["ids"] == ["good"]
    assert response["errors"][0]["id"] == "bad"


def test_reindexing_an_id_overwrites_it(api) -> None:
    seed(api)

    api.post(
        "/collections/swatches/index",
        json={
            "images": [
                {
                    "id": "red",
                    "source": {
                        "type": "path",
                        "value": str(api.image_root / "blue.png"),
                    },
                    "metadata": {"color": "recoloured"},
                }
            ]
        },
    )

    record = api.get("/collections/swatches/images/red").json()
    assert record["metadata"]["color"] == "recoloured"
    assert api.get("/collections/swatches/stats").json()["points_count"] == len(COLORS)


def test_path_outside_the_allowed_root_is_refused(api) -> None:
    api.post("/collections", json={"name": "swatches"})

    response = api.post(
        "/collections/swatches/index",
        json={
            "images": [{"id": "x", "source": {"type": "path", "value": "/etc/passwd"}}]
        },
    ).json()

    assert response["failed_count"] == 1
    assert response["errors"][0]["code"] == "IMAGE_PATH_NOT_ALLOWED"


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def test_search_by_path_ranks_the_exact_match_first(api) -> None:
    seed(api)

    body = api.post(
        "/collections/swatches/search",
        json={
            "source": {"type": "path", "value": str(api.image_root / "red.png")},
            "top_k": 4,
        },
    ).json()

    assert body["results"][0]["id"] == "red"
    assert body["results"][0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert body["results"][0]["display_image_url"] == "https://cdn.example.com/red.png"


def test_search_by_upload(api) -> None:
    seed(api)

    body = api.post(
        "/collections/swatches/search/upload",
        files={"image": ("q.png", BytesIO(png_bytes(COLORS["blue"])), "image/png")},
        data={"top_k": "2"},
    ).json()

    assert body["results"][0]["id"] == "blue"
    assert len(body["results"]) == 2


def test_search_by_text(api) -> None:
    seed(api)

    body = api.post(
        "/collections/swatches/search/text",
        json={"query": "a green swatch", "top_k": 2},
    ).json()

    assert body["results"][0]["id"] == "green"


def test_hybrid_search_shifts_with_weight(api) -> None:
    seed(api)
    query = {
        "source": {"type": "path", "value": str(api.image_root / "red.png")},
        "query": "blue",
        "top_k": 4,
    }

    image_led = api.post(
        "/collections/swatches/search/hybrid", json={**query, "text_weight": 0.0}
    ).json()
    text_led = api.post(
        "/collections/swatches/search/hybrid", json={**query, "text_weight": 1.0}
    ).json()

    assert image_led["results"][0]["id"] == "red"
    assert text_led["results"][0]["id"] == "blue"


def test_batch_search_average_and_separate(api) -> None:
    seed(api)
    sources = [
        {"type": "path", "value": str(api.image_root / "red.png")},
        {"type": "path", "value": str(api.image_root / "blue.png")},
    ]

    averaged = api.post(
        "/collections/swatches/search/batch",
        json={"sources": sources, "mode": "average", "top_k": 4},
    ).json()
    separate = api.post(
        "/collections/swatches/search/batch",
        json={"sources": sources, "mode": "separate", "top_k": 2},
    ).json()

    assert averaged["groups"] == []
    assert len(averaged["results"]) > 0
    assert [g["source_index"] for g in separate["groups"]] == [0, 1]
    assert separate["groups"][0]["results"][0]["id"] == "red"
    assert separate["groups"][1]["results"][0]["id"] == "blue"


def test_default_top_k_comes_from_settings(api) -> None:
    seed(api)

    body = api.post(
        "/collections/swatches/search",
        json={"source": {"type": "path", "value": str(api.image_root / "red.png")}},
    ).json()

    assert body["top_k"] == 5


def test_top_k_above_the_maximum_is_refused(api) -> None:
    seed(api)

    response = api.post(
        "/collections/swatches/search",
        json={
            "source": {"type": "path", "value": str(api.image_root / "red.png")},
            "top_k": 500,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "TOP_K_TOO_LARGE"


def test_min_score_filters_weak_matches(api) -> None:
    seed(api)

    body = api.post(
        "/collections/swatches/search",
        json={
            "source": {"type": "path", "value": str(api.image_root / "red.png")},
            "top_k": 4,
            "min_score": 0.99,
        },
    ).json()

    assert [r["id"] for r in body["results"]] == ["red"]


def test_metadata_filters_apply(api) -> None:
    seed(api)

    exact = api.post(
        "/collections/swatches/search",
        json={
            "source": {"type": "path", "value": str(api.image_root / "red.png")},
            "top_k": 4,
            "filters": {"color": "blue"},
        },
    ).json()
    ranged = api.post(
        "/collections/swatches/search",
        json={
            "source": {"type": "path", "value": str(api.image_root / "red.png")},
            "top_k": 4,
            "filters": {"price": {"gte": 20}},
        },
    ).json()
    listed = api.post(
        "/collections/swatches/search",
        json={
            "source": {"type": "path", "value": str(api.image_root / "red.png")},
            "top_k": 4,
            "filters": {"color": ["red", "green"]},
        },
    ).json()

    assert [r["id"] for r in exact["results"]] == ["blue"]
    assert all(r["metadata"]["price"] >= 20 for r in ranged["results"])
    assert {r["id"] for r in listed["results"]} == {"red", "green"}


def test_an_unapplicable_filter_is_refused_not_ignored(api) -> None:
    seed(api)

    response = api.post(
        "/collections/swatches/search",
        json={
            "source": {"type": "path", "value": str(api.image_root / "red.png")},
            "filters": {"price": {"around": 20}},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UNSUPPORTED_FILTER"


# --------------------------------------------------------------------------
# Image records
# --------------------------------------------------------------------------


def test_image_record_read_list_and_delete(api) -> None:
    seed(api)

    record = api.get("/collections/swatches/images/red").json()
    assert record["source_type"] == "path"
    assert record["embedding_model"] == "color-v1"
    assert record["vector_dimension"] == 3

    listing = api.get("/collections/swatches/images", params={"limit": 2}).json()
    assert len(listing["images"]) == 2

    assert api.delete("/collections/swatches/images/red").json()["deleted"] is True
    assert api.get("/collections/swatches/images/red").status_code == 404
    assert (
        api.get("/collections/swatches/stats").json()["points_count"] == len(COLORS) - 1
    )


def test_listing_pages_through_every_record(api) -> None:
    seed(api)

    seen, cursor = [], None
    for _ in range(10):
        params = {"limit": 2}
        if cursor:
            params["cursor"] = cursor
        page = api.get("/collections/swatches/images", params=params).json()
        seen.extend(image["id"] for image in page["images"])
        cursor = page["next_cursor"]
        if cursor is None:
            break

    assert sorted(seen) == sorted(COLORS)


def test_deleting_a_missing_image_is_a_404(api) -> None:
    seed(api)
    assert api.delete("/collections/swatches/images/nope").status_code == 404


# --------------------------------------------------------------------------
# Validation and limits
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/collections", {"name": ""}),
        ("/collections/swatches/search", {"source": {"type": "url", "value": "ftp://x"}}),
        (
            "/collections/swatches/search",
            {"source": {"type": "path", "value": "a"}, "top_k": 0},
        ),
        ("/collections/swatches/search/text", {"query": ""}),
        ("/collections/swatches/index", {"images": []}),
    ],
)
def test_malformed_requests_are_rejected_with_one_error_shape(api, path, payload) -> None:
    response = api.post(path, json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_oversized_upload_is_refused(api) -> None:
    seed(api)

    response = api.post(
        "/collections/swatches/search/upload",
        files={"image": ("big.png", BytesIO(b"\xff" * 5 * 1024 * 1024), "image/png")},
    )

    assert response.status_code == 413


def test_a_corrupt_upload_is_refused(api) -> None:
    seed(api)

    response = api.post(
        "/collections/swatches/search/upload",
        files={"image": ("q.png", BytesIO(b"definitely not a png"), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_unknown_route_uses_the_same_error_shape(api) -> None:
    body = api.get("/nope").json()
    assert body["error"]["code"] == "NOT_FOUND"
