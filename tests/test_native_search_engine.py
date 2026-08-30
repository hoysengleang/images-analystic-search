"""Shared correctness requirements exercised against the native engine."""

from __future__ import annotations

import math
import random
from dataclasses import replace

import pytest

from app.db import connect, migrate, transaction
from app.db.repositories import (
    ProductImageRepository,
    ProductRepository,
    SourceRepository,
    TenantRepository,
)
from app.models import Product, ProductImage, Source, Tenant, VectorRecord
from app.search_engines import NativeSearchEngine, VectorSearchRequest

MODEL = "test:color@1"


@pytest.fixture
def native_catalogue(db):
    with transaction(db):
        tenants = TenantRepository(db)
        for tenant_id in ("shop-a", "shop-b"):
            tenants.create(Tenant(id=tenant_id, name=tenant_id))
            SourceRepository(db, tenant_id=tenant_id).create(
                Source(
                    id="catalogue",
                    tenant_id=tenant_id,
                    type="test",
                    name="Catalogue",
                )
            )

        products = ProductRepository(db, tenant_id="shop-a")
        images = ProductImageRepository(db, tenant_id="shop-a")
        data = [
            ("red-shoe", "shoes", 50.0, True, "red", [1.0, 0.0, 0.0]),
            ("red-bag", "bags", 150.0, True, "red", [0.9, 0.1, 0.0]),
            ("blue-shoe", "shoes", 80.0, False, "blue", [0.0, 0.0, 1.0]),
        ]
        records = []
        for product_id, category, price, stock, color, vector in data:
            products.upsert(
                Product(
                    id=product_id,
                    tenant_id="shop-a",
                    source_id="catalogue",
                    external_id=product_id,
                    content_hash=f"hash:{product_id}",
                    category=category,
                    price=price,
                    in_stock=stock,
                    attributes={"appearance": {"color": color}},
                )
            )
            image_id = f"image-{product_id}"
            images.upsert(
                ProductImage(
                    id=image_id,
                    tenant_id="shop-a",
                    product_id=product_id,
                    source_uri=f"images/{product_id}.jpg",
                )
            )
            records.append(
                VectorRecord(
                    id=f"vector-{product_id}",
                    tenant_id="shop-a",
                    product_id=product_id,
                    image_id=image_id,
                    model_version=MODEL,
                    dimension=3,
                    vector=vector,
                )
            )

        other_products = ProductRepository(db, tenant_id="shop-b")
        other_images = ProductImageRepository(db, tenant_id="shop-b")
        other_products.upsert(
            Product(
                id="secret",
                tenant_id="shop-b",
                source_id="catalogue",
                external_id="secret",
                content_hash="hash:secret",
            )
        )
        other_images.upsert(
            ProductImage(
                id="image-secret",
                tenant_id="shop-b",
                product_id="secret",
                source_uri="images/secret.jpg",
            )
        )

    engine = NativeSearchEngine(db)
    engine.upsert(records)
    engine.upsert(
        [
            VectorRecord(
                id="vector-secret",
                tenant_id="shop-b",
                product_id="secret",
                image_id="image-secret",
                model_version=MODEL,
                dimension=3,
                vector=[1.0, 0.0, 0.0],
            )
        ]
    )
    return engine


def request(**overrides) -> VectorSearchRequest:
    values = {
        "tenant_id": "shop-a",
        "model_version": MODEL,
        "vector": [1.0, 0.0, 0.0],
        "limit": 10,
    }
    values.update(overrides)
    return VectorSearchRequest(**values)


def test_exact_search_returns_best_cosine_match_first(native_catalogue) -> None:
    hits = native_catalogue.search(request())

    assert [hit.product_id for hit in hits] == ["red-shoe", "red-bag", "blue-shoe"]
    assert hits[0].score == pytest.approx(1.0)


def test_search_never_leaks_another_tenant(native_catalogue) -> None:
    hits = native_catalogue.search(request())

    assert "secret" not in {hit.product_id for hit in hits}


def test_search_never_mixes_model_versions(native_catalogue) -> None:
    hits = native_catalogue.search(request(model_version="test:color@old"))

    assert hits == []


def test_upsert_refuses_dimensions_mixed_under_one_model(native_catalogue) -> None:
    with pytest.raises(ValueError, match="cannot mix vector dimensions"):
        native_catalogue.upsert(
            [
                VectorRecord(
                    id="wrong-dimension",
                    tenant_id="shop-a",
                    product_id="red-shoe",
                    image_id="image-red-shoe",
                    model_version=MODEL,
                    dimension=2,
                    vector=[1.0, 0.0],
                )
            ]
        )

    assert native_catalogue.search(request())[0].product_id == "red-shoe"


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"category": "shoes"}, {"red-shoe", "blue-shoe"}),
        ({"in_stock": True, "price": {"lt": 100}}, {"red-shoe"}),
        ({"category": ["bags"]}, {"red-bag"}),
        ({"attributes.appearance.color": "blue"}, {"blue-shoe"}),
    ],
)
def test_product_filters_are_applied(native_catalogue, filters, expected) -> None:
    found = {hit.product_id for hit in native_catalogue.search(request(filters=filters))}

    assert found == expected


def test_unsupported_filters_are_refused_even_for_an_empty_model(
    native_catalogue,
) -> None:
    with pytest.raises(ValueError, match="Unsupported native-engine filter"):
        native_catalogue.search(
            request(model_version="missing", filters={"not_a_product_field": "x"})
        )


def test_products_can_be_excluded(native_catalogue) -> None:
    hits = native_catalogue.search(request(exclude_product_ids=("red-shoe",)))

    assert [hit.product_id for hit in hits] == ["red-bag", "blue-shoe"]


def test_deleted_products_do_not_appear(native_catalogue) -> None:
    ProductRepository(native_catalogue.connection, tenant_id="shop-a").mark_deleted(
        "red-shoe"
    )

    assert "red-shoe" not in {
        hit.product_id for hit in native_catalogue.search(request())
    }


def test_upsert_refreshes_an_already_loaded_cache(native_catalogue) -> None:
    assert native_catalogue.search(request())[0].product_id == "red-shoe"
    replacement = VectorRecord(
        id="vector-blue-shoe",
        tenant_id="shop-a",
        product_id="blue-shoe",
        image_id="image-blue-shoe",
        model_version=MODEL,
        dimension=3,
        vector=[1.0, 0.0, 0.0],
    )

    native_catalogue.upsert([replacement])

    assert native_catalogue.search(request())[0].product_id == "blue-shoe"


def test_delete_operations_are_tenant_scoped(native_catalogue) -> None:
    native_catalogue.delete("shop-a", ["vector-red-shoe"])
    native_catalogue.delete_for_product("shop-a", "red-bag")

    assert [hit.product_id for hit in native_catalogue.search(request())] == ["blue-shoe"]
    assert [
        hit.product_id
        for hit in native_catalogue.search(
            request(tenant_id="shop-b", exclude_product_ids=())
        )
    ] == ["secret"]


def test_vectors_survive_engine_recreation(native_catalogue) -> None:
    restarted = NativeSearchEngine(native_catalogue.connection)

    assert restarted.search(request())[0].product_id == "red-shoe"


def test_vectors_survive_database_reconnection(native_catalogue, tmp_path) -> None:
    database_path = tmp_path / "persistent.db"
    connection = connect(database_path)
    migrate(connection)
    # Reuse the fully tested fixture data by backing up its SQLite connection.
    native_catalogue.connection.backup(connection)
    connection.close()

    reopened = connect(database_path)
    try:
        restarted = NativeSearchEngine(reopened)
        assert restarted.search(request())[0].product_id == "red-shoe"
    finally:
        reopened.close()


@pytest.mark.parametrize("vector", [[], [0.0, 0.0, 0.0], [1.0, float("nan"), 0.0]])
def test_invalid_query_vectors_are_refused(native_catalogue, vector) -> None:
    if not vector:
        with pytest.raises(ValueError):
            replace(request(), vector=vector)
        return

    with pytest.raises(ValueError):
        native_catalogue.search(request(vector=vector))


def test_health_checks_the_database(native_catalogue) -> None:
    assert native_catalogue.health().healthy is True


@pytest.mark.parametrize("seed", [20260830, 90210, 8675309])
def test_randomized_ranking_matches_an_independent_cosine_oracle(db, seed) -> None:
    """Exercise exact top-k ordering beyond a few hand-picked directions."""
    generator = random.Random(seed)
    tenant_id = "oracle-shop"
    dimension = 11
    model_version = "test:random@1"
    catalogue_size = 240

    with transaction(db):
        TenantRepository(db).create(Tenant(id=tenant_id, name="Oracle Shop"))
        SourceRepository(db, tenant_id=tenant_id).create(
            Source(
                id="catalogue",
                tenant_id=tenant_id,
                type="test",
                name="Catalogue",
            )
        )
        products = ProductRepository(db, tenant_id=tenant_id)
        images = ProductImageRepository(db, tenant_id=tenant_id)
        records = []
        for index in range(catalogue_size):
            product_id = f"product-{index:04d}"
            image_id = f"image-{index:04d}"
            vector = [generator.uniform(-3.0, 3.0) for _ in range(dimension)]
            products.upsert(
                Product(
                    id=product_id,
                    tenant_id=tenant_id,
                    source_id="catalogue",
                    external_id=product_id,
                    content_hash=f"hash:{index}",
                )
            )
            images.upsert(
                ProductImage(
                    id=image_id,
                    tenant_id=tenant_id,
                    product_id=product_id,
                    source_uri=f"images/{index}.jpg",
                )
            )
            records.append(
                VectorRecord(
                    id=f"vector-{index:04d}",
                    tenant_id=tenant_id,
                    product_id=product_id,
                    image_id=image_id,
                    model_version=model_version,
                    dimension=dimension,
                    vector=vector,
                    normalized=False,
                )
            )

    engine = NativeSearchEngine(db)
    engine.upsert(records)

    def cosine(left, right) -> float:
        numerator = math.fsum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(math.fsum(value * value for value in left))
        right_norm = math.sqrt(math.fsum(value * value for value in right))
        return numerator / (left_norm * right_norm)

    for _ in range(30):
        query = [generator.uniform(-2.0, 2.0) for _ in range(dimension)]
        limit = generator.randint(1, 40)
        expected = sorted(
            ((record.id, cosine(query, record.vector)) for record in records),
            key=lambda item: (-item[1], item[0]),
        )[:limit]

        actual = engine.search(
            VectorSearchRequest(
                tenant_id=tenant_id,
                model_version=model_version,
                vector=query,
                limit=limit,
            )
        )

        assert [hit.vector_id for hit in actual] == [item[0] for item in expected]
        assert [hit.score for hit in actual] == pytest.approx(
            [item[1] for item in expected], abs=2e-7
        )
