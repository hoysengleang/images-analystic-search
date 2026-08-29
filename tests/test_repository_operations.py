"""Repository operations Milestone 1's synchronisation depends on.

These were written before their caller exists, so they are covered here rather
than left as untested code waiting to surprise someone.
"""

from datetime import datetime, timezone

import pytest

from app.db import transaction
from app.db.repositories import (
    ApiKeyRepository,
    ProductImageRepository,
    ProductRepository,
    SourceRepository,
    TenantRepository,
    VectorRepository,
)
from app.db.repositories.vectors import pack_vector, unpack_vector
from app.models import (
    ApiKey,
    EmbeddingStatus,
    Product,
    ProductImage,
    Source,
    SourceStatus,
    Tenant,
    VectorRecord,
)

MODEL_VERSION = "openclip:ViT-B-32@laion2b_s34b_b79k"


@pytest.fixture
def catalogue(db):
    with transaction(db):
        TenantRepository(db).create(Tenant(id="t1", name="Shop"))
        SourceRepository(db, tenant_id="t1").create(
            Source(id="s1", tenant_id="t1", type="local_manifest", name="Main")
        )
        products = ProductRepository(db, tenant_id="t1")
        images = ProductImageRepository(db, tenant_id="t1")
        for index in range(3):
            products.upsert(
                Product(
                    id=f"p{index}",
                    tenant_id="t1",
                    source_id="s1",
                    external_id=f"SKU-{index}",
                    content_hash=f"sha256:{index}",
                    title=f"Product {index}",
                )
            )
            images.upsert(
                ProductImage(
                    id=f"i{index}",
                    tenant_id="t1",
                    product_id=f"p{index}",
                    source_uri=f"images/{index}.jpg",
                )
            )
    return db


def test_content_hashes_drive_skipping_unchanged_products(catalogue) -> None:
    hashes = ProductRepository(catalogue, tenant_id="t1").content_hashes("s1")

    assert hashes == {"SKU-0": "sha256:0", "SKU-1": "sha256:1", "SKU-2": "sha256:2"}


def test_deleted_products_drop_out_of_the_hash_map(catalogue) -> None:
    products = ProductRepository(catalogue, tenant_id="t1")
    products.mark_deleted("p1")

    assert "SKU-1" not in products.content_hashes("s1")


def test_pending_images_are_the_work_queue(catalogue) -> None:
    images = ProductImageRepository(catalogue, tenant_id="t1")

    assert len(images.list_pending()) == 3

    images.set_status("i0", status=EmbeddingStatus.READY, model_version=MODEL_VERSION)

    pending = images.list_pending()
    assert [image.id for image in pending] == ["i1", "i2"]
    assert images.get("i0").embedding_status is EmbeddingStatus.READY
    assert images.get("i0").model_version == MODEL_VERSION


def test_a_failed_image_records_why(catalogue) -> None:
    images = ProductImageRepository(catalogue, tenant_id="t1")

    images.set_status("i1", status=EmbeddingStatus.FAILED, error="404 from origin")

    failed = images.get("i1")
    assert failed.embedding_status is EmbeddingStatus.FAILED
    assert failed.error == "404 from origin"
    assert failed not in images.list_pending()


def test_pending_listing_is_capped(catalogue) -> None:
    assert (
        len(ProductImageRepository(catalogue, tenant_id="t1").list_pending(limit=2)) == 2
    )


def test_sync_state_round_trips(catalogue) -> None:
    sources = SourceRepository(catalogue, tenant_id="t1")
    synced_at = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)

    sources.update_sync_state(
        "s1", status=SourceStatus.SYNCING, sync_cursor="page-2", last_synced_at=synced_at
    )

    source = sources.get("s1")
    assert source.status is SourceStatus.SYNCING
    assert source.sync_cursor == "page-2"
    assert source.last_synced_at == synced_at


def test_listing_a_tenants_api_keys(db) -> None:
    with transaction(db):
        TenantRepository(db).create(Tenant(id="t1", name="Shop"))
        TenantRepository(db).create(Tenant(id="t2", name="Other"))
        keys = ApiKeyRepository(db)
        keys.create(ApiKey(id="k1", tenant_id="t1", name="first", key_hash="h1"))
        keys.create(ApiKey(id="k2", tenant_id="t1", name="second", key_hash="h2"))
        keys.create(ApiKey(id="k3", tenant_id="t2", name="other", key_hash="h3"))

    assert [key.id for key in ApiKeyRepository(db).list_for_tenant("t1")] == ["k1", "k2"]


def test_replacing_one_images_vector(catalogue) -> None:
    """A changed image replaces its vector without touching its siblings."""
    vectors = VectorRepository(catalogue, tenant_id="t1")
    with transaction(catalogue):
        vectors.upsert_many(
            [
                VectorRecord(
                    id=f"v{index}",
                    tenant_id="t1",
                    product_id=f"p{index}",
                    image_id=f"i{index}",
                    model_version=MODEL_VERSION,
                    dimension=3,
                    vector=[0.1 * index, 0.2, 0.3],
                )
                for index in range(3)
            ]
        )

    assert vectors.delete_for_image("i1") == 1
    assert vectors.count() == 2
    assert vectors.get("v1") is None
    assert vectors.get("v0") is not None


def test_deleting_every_vector_for_a_product(catalogue) -> None:
    vectors = VectorRepository(catalogue, tenant_id="t1")
    with transaction(catalogue):
        vectors.upsert_many(
            [
                VectorRecord(
                    id=f"v{n}",
                    tenant_id="t1",
                    product_id="p0",
                    image_id="i0",
                    model_version=MODEL_VERSION,
                    dimension=3,
                    vector=[0.1, 0.2, 0.3],
                )
                for n in range(3)
            ]
        )

    assert vectors.delete_for_product("p0") == 3
    assert vectors.count() == 0


def test_vectors_are_counted_per_model_version(catalogue) -> None:
    vectors = VectorRepository(catalogue, tenant_id="t1")
    with transaction(catalogue):
        vectors.upsert_many(
            [
                VectorRecord(
                    id="old",
                    tenant_id="t1",
                    product_id="p0",
                    image_id="i0",
                    model_version="openclip:ViT-B-32@v1",
                    dimension=3,
                    vector=[1.0, 0.0, 0.0],
                ),
                VectorRecord(
                    id="new",
                    tenant_id="t1",
                    product_id="p1",
                    image_id="i1",
                    model_version=MODEL_VERSION,
                    dimension=3,
                    vector=[0.0, 1.0, 0.0],
                ),
            ]
        )

    assert vectors.count() == 2
    assert vectors.count(MODEL_VERSION) == 1
    assert [v.id for v in vectors.list_for_model(MODEL_VERSION)] == ["new"]


def test_upsert_replaces_a_vector_in_place(catalogue) -> None:
    vectors = VectorRepository(catalogue, tenant_id="t1")
    record = VectorRecord(
        id="v0",
        tenant_id="t1",
        product_id="p0",
        image_id="i0",
        model_version=MODEL_VERSION,
        dimension=3,
        vector=[1.0, 0.0, 0.0],
    )
    with transaction(catalogue):
        vectors.upsert_many([record])
        vectors.upsert_many(
            [
                VectorRecord(
                    id="v0",
                    tenant_id="t1",
                    product_id="p0",
                    image_id="i0",
                    model_version=MODEL_VERSION,
                    dimension=3,
                    vector=[0.0, 0.0, 1.0],
                )
            ]
        )

    assert vectors.count() == 1
    assert vectors.get("v0").vector == pytest.approx([0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "values",
    [[0.0], [1.0, -1.0], [0.5] * 512, [1e-8, 1e8, -3.25]],
)
def test_vector_blobs_round_trip_exactly(values) -> None:
    assert unpack_vector(pack_vector(values)) == pytest.approx(values, rel=1e-6)


def test_a_blob_is_four_bytes_per_dimension() -> None:
    assert len(pack_vector([0.1] * 512)) == 512 * 4
