"""Tenant isolation is priority #1 in the specification.

These tests treat it as a storage-layer property: a repository cannot be
built without a tenant, and no query may cross one.
"""

import sqlite3

import pytest

from app.db import transaction
from app.db.repositories import (
    ProductImageRepository,
    ProductRepository,
    SourceRepository,
    TenantRepository,
    VectorRepository,
)
from app.models import Product, ProductImage, Source, Tenant, VectorRecord

MODEL_VERSION = "openclip:ViT-B-32@laion2b_s34b_b79k"

SCOPED_REPOSITORIES = [
    ProductRepository,
    ProductImageRepository,
    SourceRepository,
    VectorRepository,
]


@pytest.fixture
def two_tenants(db):
    """Two tenants, each with a source, a product, an image, and a vector."""
    with transaction(db):
        tenants = TenantRepository(db)
        for tenant_id, name in (("t1", "Shop One"), ("t2", "Shop Two")):
            tenants.create(Tenant(id=tenant_id, name=name))
            SourceRepository(db, tenant_id=tenant_id).create(
                Source(
                    id=f"src_{tenant_id}",
                    tenant_id=tenant_id,
                    type="local_manifest",
                    name="Main catalogue",
                )
            )
            ProductRepository(db, tenant_id=tenant_id).upsert(
                Product(
                    id=f"prod_{tenant_id}",
                    tenant_id=tenant_id,
                    source_id=f"src_{tenant_id}",
                    external_id="SHOE-1",
                    content_hash="sha256:aaa",
                    title=f"{name} shoe",
                    category="shoes",
                )
            )
            ProductImageRepository(db, tenant_id=tenant_id).upsert(
                ProductImage(
                    id=f"img_{tenant_id}",
                    tenant_id=tenant_id,
                    product_id=f"prod_{tenant_id}",
                    source_uri="images/a.jpg",
                )
            )
            VectorRepository(db, tenant_id=tenant_id).upsert_many(
                [
                    VectorRecord(
                        id=f"vec_{tenant_id}",
                        tenant_id=tenant_id,
                        product_id=f"prod_{tenant_id}",
                        image_id=f"img_{tenant_id}",
                        model_version=MODEL_VERSION,
                        dimension=3,
                        vector=[0.1, 0.2, 0.3],
                    )
                ]
            )
    return db


@pytest.mark.parametrize("repository_class", SCOPED_REPOSITORIES)
@pytest.mark.parametrize("tenant_id", ["", "   ", None])
def test_a_scoped_repository_cannot_be_built_without_a_tenant(
    db, repository_class, tenant_id
) -> None:
    with pytest.raises((ValueError, AttributeError, TypeError)):
        repository_class(db, tenant_id=tenant_id)


def test_a_product_is_invisible_to_another_tenant(two_tenants) -> None:
    assert ProductRepository(two_tenants, tenant_id="t2").get("prod_t1") is None


def test_listing_never_leaks_another_tenants_products(two_tenants) -> None:
    titles = [p.title for p in ProductRepository(two_tenants, tenant_id="t2").list()]

    assert titles == ["Shop Two shoe"]


def test_bulk_lookup_drops_ids_belonging_to_another_tenant(two_tenants) -> None:
    found = ProductRepository(two_tenants, tenant_id="t2").get_many(
        ["prod_t1", "prod_t2"]
    )

    assert list(found) == ["prod_t2"]


def test_external_id_lookup_is_scoped(two_tenants) -> None:
    """Both tenants use the same external id; each must see only its own."""
    found = ProductRepository(two_tenants, tenant_id="t2").get_by_external_id(
        "src_t1", "SHOE-1"
    )

    assert found is None


def test_a_source_is_invisible_to_another_tenant(two_tenants) -> None:
    assert SourceRepository(two_tenants, tenant_id="t2").get("src_t1") is None


def test_an_image_is_invisible_to_another_tenant(two_tenants) -> None:
    assert ProductImageRepository(two_tenants, tenant_id="t2").get("img_t1") is None


def test_a_vector_is_invisible_to_another_tenant(two_tenants) -> None:
    repository = VectorRepository(two_tenants, tenant_id="t2")

    assert repository.get("vec_t1") is None
    assert repository.count() == 1
    assert [v.id for v in repository.list_for_model(MODEL_VERSION)] == ["vec_t2"]


def test_deleting_another_tenants_data_is_a_no_op(two_tenants) -> None:
    ProductRepository(two_tenants, tenant_id="t2").delete("prod_t1")
    VectorRepository(two_tenants, tenant_id="t2").delete(["vec_t1"])

    assert ProductRepository(two_tenants, tenant_id="t1").get("prod_t1") is not None
    assert VectorRepository(two_tenants, tenant_id="t1").get("vec_t1") is not None


def test_writing_another_tenants_record_is_refused(two_tenants) -> None:
    """A mismatched tenant on the object is a bug, not a silent rewrite."""
    with pytest.raises(ValueError):
        ProductRepository(two_tenants, tenant_id="t2").upsert(
            Product(
                id="smuggled",
                tenant_id="t1",
                source_id="src_t1",
                external_id="SHOE-9",
                content_hash="h",
            )
        )


def test_marking_deleted_does_not_reach_another_tenant(two_tenants) -> None:
    ProductRepository(two_tenants, tenant_id="t2").mark_deleted("prod_t1")

    assert not ProductRepository(two_tenants, tenant_id="t1").get("prod_t1").is_deleted


def test_product_cannot_reference_another_tenants_source(two_tenants) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        ProductRepository(two_tenants, tenant_id="t2").upsert(
            Product(
                id="cross-source",
                tenant_id="t2",
                source_id="src_t1",
                external_id="SHOE-9",
                content_hash="h",
            )
        )


def test_image_cannot_reference_another_tenants_product(two_tenants) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        ProductImageRepository(two_tenants, tenant_id="t2").upsert(
            ProductImage(
                id="cross-product",
                tenant_id="t2",
                product_id="prod_t1",
                source_uri="images/cross.jpg",
            )
        )


def test_vector_cannot_reference_another_tenants_image(two_tenants) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        VectorRepository(two_tenants, tenant_id="t2").upsert_many(
            [
                VectorRecord(
                    id="cross-image",
                    tenant_id="t2",
                    product_id="prod_t1",
                    image_id="img_t1",
                    model_version=MODEL_VERSION,
                    dimension=3,
                    vector=[1.0, 0.0, 0.0],
                )
            ]
        )


def test_same_vector_id_is_isolated_per_tenant(two_tenants) -> None:
    VectorRepository(two_tenants, tenant_id="t1").upsert_many(
        [
            VectorRecord(
                id="shared-vector-id",
                tenant_id="t1",
                product_id="prod_t1",
                image_id="img_t1",
                model_version=MODEL_VERSION,
                dimension=3,
                vector=[1.0, 0.0, 0.0],
            )
        ]
    )
    VectorRepository(two_tenants, tenant_id="t2").upsert_many(
        [
            VectorRecord(
                id="shared-vector-id",
                tenant_id="t2",
                product_id="prod_t2",
                image_id="img_t2",
                model_version=MODEL_VERSION,
                dimension=3,
                vector=[0.0, 1.0, 0.0],
            )
        ]
    )

    tenant_one = VectorRepository(two_tenants, tenant_id="t1").get("shared-vector-id")
    tenant_two = VectorRepository(two_tenants, tenant_id="t2").get("shared-vector-id")
    assert tenant_one.vector == pytest.approx([1.0, 0.0, 0.0])
    assert tenant_two.vector == pytest.approx([0.0, 1.0, 0.0])
