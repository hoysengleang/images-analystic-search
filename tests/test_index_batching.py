from pathlib import Path

from PIL import Image

from app.core.config import Settings
from app.schemas.collection import CollectionModelConfig
from app.schemas.index import IndexRequest
from app.services.collection_metadata_service import CollectionMetadataService
from app.services.indexing_service import IndexingService


class FakeImageLoader:
    def load_from_source(self, source):
        return Image.new("RGB", (2, 2))


class RecordingEmbeddingProvider:
    def __init__(self) -> None:
        self.batch_sizes: list = []

    def embed_images(self, images):
        self.batch_sizes.append(len(images))
        return [[1.0, 0.0, 0.0] for _ in images]


class FakeEmbeddingManager:
    def __init__(self, provider) -> None:
        self.provider = provider

    def get_provider(self, **_kwargs):
        return self.provider


class RecordingVectorService:
    def __init__(self) -> None:
        self.upsert_calls: list = []

    def upsert_images(self, *, collection_name, records, embedding):
        self.upsert_calls.append([record.image_id for record in records])
        return [record.image_id for record in records]


def make_service(tmp_path: Path, *, embed_batch_size: int):
    metadata_service = CollectionMetadataService(tmp_path / "collections.json")
    metadata_service.save(
        collection_name="products",
        model=CollectionModelConfig(
            provider="openclip",
            name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            vector_size=3,
            distance="cosine",
        ),
    )
    provider = RecordingEmbeddingProvider()
    vector_service = RecordingVectorService()
    service = IndexingService(
        settings=Settings(embed_batch_size=embed_batch_size),
        metadata_service=metadata_service,
        image_loader=FakeImageLoader(),
        embedding_manager=FakeEmbeddingManager(provider),
        vector_service=vector_service,
    )
    return service, provider, vector_service


def index_request(count: int) -> IndexRequest:
    return IndexRequest(
        collection_name="products",
        images=[
            {
                "id": f"img_{index:03d}",
                "source": {
                    "type": "url",
                    "value": f"https://cdn.example.com/{index}.jpg",
                },
            }
            for index in range(count)
        ],
    )


def test_images_are_embedded_in_batches(tmp_path: Path) -> None:
    service, provider, _ = make_service(tmp_path, embed_batch_size=4)

    response = service.index(index_request(10))

    assert response.indexed_count == 10
    assert provider.batch_sizes == [4, 4, 2]


def test_each_batch_is_stored_in_one_round_trip(tmp_path: Path) -> None:
    service, _, vector_service = make_service(tmp_path, embed_batch_size=4)

    service.index(index_request(10))

    assert [len(call) for call in vector_service.upsert_calls] == [4, 4, 2]


def test_a_failing_batch_does_not_lose_the_other_batches(tmp_path: Path) -> None:
    service, provider, vector_service = make_service(tmp_path, embed_batch_size=2)
    original_embed_images = provider.embed_images

    def fail_on_second_batch(images):
        if len(provider.batch_sizes) == 1:
            provider.batch_sizes.append(len(images))
            raise RuntimeError("cuda out of memory")
        return original_embed_images(images)

    provider.embed_images = fail_on_second_batch

    response = service.index(index_request(6))

    assert response.indexed_count == 4
    assert response.failed_count == 2
    assert {error["code"] for error in response.errors} == {"INDEX_IMAGE_FAILED"}
    assert response.ids == ["img_000", "img_001", "img_004", "img_005"]


class CountingImageLoader:
    """Tracks how many images have been decoded so far."""

    def __init__(self) -> None:
        self.loaded = 0

    def load_from_source(self, source):
        self.loaded += 1
        return Image.new("RGB", (2, 2))


class LoadObservingProvider:
    """Records how many images were decoded before each embed call."""

    def __init__(self, loader: CountingImageLoader) -> None:
        self.loader = loader
        self.loaded_at_embed: list = []

    def embed_images(self, images):
        self.loaded_at_embed.append(self.loader.loaded)
        return [[1.0, 0.0, 0.0] for _ in images]


def test_images_are_loaded_lazily_one_batch_ahead(tmp_path: Path) -> None:
    """Decoding every image up front would blow memory on a large catalogue."""
    metadata_service = CollectionMetadataService(tmp_path / "collections.json")
    metadata_service.save(
        collection_name="products",
        model=CollectionModelConfig(
            provider="openclip",
            name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            vector_size=3,
            distance="cosine",
        ),
    )
    loader = CountingImageLoader()
    provider = LoadObservingProvider(loader)
    service = IndexingService(
        settings=Settings(embed_batch_size=4),
        metadata_service=metadata_service,
        image_loader=loader,
        embedding_manager=FakeEmbeddingManager(provider),
        vector_service=RecordingVectorService(),
    )

    service.index(index_request(20))

    # Each embed call sees only the images of its own batch already decoded,
    # so peak memory tracks EMBED_BATCH_SIZE rather than the request size.
    assert provider.loaded_at_embed == [4, 8, 12, 16, 20]
