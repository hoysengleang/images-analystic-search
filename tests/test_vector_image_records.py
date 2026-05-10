from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.embedding.base import EmbeddingModelMetadata
from app.schemas.image import ImageSource
from app.services.vector_service import VectorService


def test_get_and_delete_image_record_from_qdrant() -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="products",
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )
    service = VectorService(qdrant_client=client)
    embedding = EmbeddingModelMetadata(
        provider="openclip",
        model_name="ViT-B-32",
        model_pretrained="laion2b_s34b_b79k",
        vector_size=3,
    )
    source = ImageSource(type="url", value="https://example.com/a.jpg")
    service.upsert_image(
        collection_name="products",
        image_id="img_001",
        vector=[1.0, 0.0, 0.0],
        source=source,
        metadata={"name": "Blue Shoe"},
        embedding=embedding,
    )

    record = service.get_image(collection_name="products", image_id="img_001")
    deleted = service.delete_image_record(
        collection_name="products",
        image_id="img_001",
    )

    assert record.id == "img_001"
    assert record.source_type == "url"
    assert record.source_value == "https://example.com/a.jpg"
    assert record.metadata == {"name": "Blue Shoe"}
    assert record.embedding_provider == "openclip"
    assert record.embedding_model == "ViT-B-32"
    assert deleted.deleted is True
    assert service.count_images(collection_name="products") == 0
