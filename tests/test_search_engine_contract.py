"""Invariants of the SearchEngine interface itself.

The shared behavioural suite that both the native engine and the Qdrant
adapter must pass arrives with the native engine in Milestone 1. What is
testable now is that the contract makes an unsafe query impossible to build.
"""

import dataclasses

import pytest

from app.search_engines import EngineHealth, SearchEngine, VectorHit, VectorSearchRequest


def valid_request(**overrides) -> dict:
    request = {
        "tenant_id": "t1",
        "model_version": "openclip:ViT-B-32@laion2b",
        "vector": [0.1, 0.2, 0.3],
        "limit": 10,
    }
    request.update(overrides)
    return request


def test_a_well_formed_request_is_accepted() -> None:
    request = VectorSearchRequest(**valid_request())

    assert request.tenant_id == "t1"
    assert request.filters == {}
    assert request.exclude_product_ids == ()


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"tenant_id": ""}, "an unscoped search must be impossible"),
        ({"tenant_id": "   "}, "whitespace is not a tenant"),
        ({"model_version": ""}, "vectors from unknown models cannot be compared"),
        ({"vector": []}, "a search needs a query vector"),
        ({"limit": 0}, "a search needs a positive limit"),
        ({"limit": -5}, "a search needs a positive limit"),
    ],
)
def test_unsafe_requests_cannot_be_constructed(overrides, reason) -> None:
    with pytest.raises(ValueError):
        VectorSearchRequest(**valid_request(**overrides))


def test_requests_are_immutable() -> None:
    request = VectorSearchRequest(**valid_request())

    with pytest.raises(dataclasses.FrozenInstanceError):
        request.tenant_id = "t2"


def test_the_engine_interface_cannot_be_partially_implemented() -> None:
    class HalfBuiltEngine(SearchEngine):
        name = "half-built"

        def upsert(self, records):
            return None

    with pytest.raises(TypeError):
        HalfBuiltEngine()


def test_a_complete_implementation_satisfies_the_interface() -> None:
    class StubEngine(SearchEngine):
        name = "stub"

        def upsert(self, records):
            return None

        def delete(self, tenant_id, vector_ids):
            return None

        def delete_for_product(self, tenant_id, product_id):
            return None

        def search(self, request):
            return [VectorHit(vector_id="v1", product_id="p1", image_id="i1", score=0.9)]

        def health(self):
            return EngineHealth(name=self.name, healthy=True, vector_count=1)

    engine = StubEngine()

    assert engine.health().healthy is True
    assert engine.search(VectorSearchRequest(**valid_request()))[0].product_id == "p1"
