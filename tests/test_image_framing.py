"""Framing decides what the model actually sees, and is pinned per collection.

The stock CLIP transform resizes the short side then centre-crops, which
discards the top and bottom of a portrait photo. Padding keeps the whole frame.
Because the two produce different vectors, a collection must keep whichever it
was built with.
"""

import pytest
from PIL import Image
from pydantic import ValidationError

from app.core.config import Settings
from app.embedding.base import EmbeddingProvider
from app.embedding.manager import EmbeddingManager
from app.embedding.registry import EmbeddingProviderRegistry
from app.schemas.collection import CollectionModelConfig
from app.services.collection_metadata_service import CollectionMetadataService
from app.utils.image_utils import center_crop_fraction, letterbox_to_square


class FramingAwareProvider(EmbeddingProvider):
    provider_name = "framing-aware"

    def __init__(
        self, *, model_name, model_pretrained, vector_size, framing="crop", views=1
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_pretrained=model_pretrained,
            vector_size=vector_size,
        )
        self.framing = framing
        self.views = views

    def embed_image(self, image) -> list:
        return [1.0, 0.0, 0.0]


class PlainProvider(EmbeddingProvider):
    """A provider that knows nothing about framing."""

    provider_name = "plain"

    def embed_image(self, image) -> list:
        return [1.0, 0.0, 0.0]


# --- letterboxing ----------------------------------------------------------


def test_padding_keeps_the_whole_frame() -> None:
    """A tall image must survive with its top and bottom intact."""
    photo = Image.new("RGB", (300, 500), (255, 255, 255))
    photo.paste(Image.new("RGB", (300, 60), (255, 0, 0)), (0, 0))
    photo.paste(Image.new("RGB", (300, 60), (0, 0, 255)), (0, 440))

    padded = letterbox_to_square(photo, 224)
    colours = {pixel for pixel in padded.getdata()}

    assert padded.size == (224, 224)
    assert any(r > 200 and g < 80 for r, g, b in colours), "top band was lost"
    assert any(b > 200 and r < 80 for r, g, b in colours), "bottom band was lost"


def test_padding_a_square_image_changes_nothing_but_scale() -> None:
    square = Image.new("RGB", (400, 400), (10, 120, 200))

    padded = letterbox_to_square(square, 224)

    assert padded.size == (224, 224)
    # No bars: every pixel is still the original colour.
    assert set(padded.getdata()) == {(10, 120, 200)}


@pytest.mark.parametrize("size", [(1, 900), (900, 1), (1, 1), (4000, 12), (12, 4000)])
def test_padding_survives_extreme_aspect_ratios(size) -> None:
    assert letterbox_to_square(Image.new("RGB", size, (5, 5, 5)), 224).size == (224, 224)


def test_padding_preserves_aspect_ratio() -> None:
    padded = letterbox_to_square(Image.new("RGB", (400, 200), (200, 0, 0)), 224)

    # A 2:1 image scaled to fit gives a 224x112 band centred vertically.
    non_background_rows = {
        y
        for y in range(224)
        if any(padded.getpixel((x, y)) == (200, 0, 0) for x in range(0, 224, 8))
    }
    assert 100 <= len(non_background_rows) <= 120


def test_center_crop_fraction_takes_the_middle() -> None:
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    image.paste(Image.new("RGB", (20, 20), (255, 255, 255)), (40, 40))

    cropped = center_crop_fraction(image, 0.5)

    assert cropped.size == (50, 50)
    assert cropped.getpixel((25, 25)) == (255, 255, 255)


# --- framing is part of a collection's identity ----------------------------


def make_manager(framing="pad", views=1, provider=FramingAwareProvider):
    registry = EmbeddingProviderRegistry()
    registry.register(provider)
    settings = Settings(
        default_embedding_provider=provider.provider_name,
        default_model_name="m",
        default_model_pretrained="none",
        default_vector_size=3,
        image_framing=framing,
        embed_views=views,
    )
    return EmbeddingManager(settings=settings, registry=registry)


def test_the_manager_passes_framing_to_providers_that_accept_it() -> None:
    provider = make_manager().get_provider(
        provider_name="framing-aware",
        model_name="m",
        model_pretrained="none",
        vector_size=3,
        framing="pad",
        views=3,
    )

    assert provider.framing == "pad"
    assert provider.views == 3


def test_providers_without_framing_are_built_anyway() -> None:
    """A provider that never heard of framing must not break."""
    provider = make_manager(provider=PlainProvider).get_provider(
        provider_name="plain",
        model_name="m",
        model_pretrained="none",
        vector_size=3,
        framing="pad",
        views=3,
    )

    assert isinstance(provider, PlainProvider)


def test_different_framings_are_different_providers() -> None:
    """Sharing one instance would silently mix two vector spaces."""
    manager = make_manager()
    common = {
        "provider_name": "framing-aware",
        "model_name": "m",
        "model_pretrained": "none",
        "vector_size": 3,
    }

    padded = manager.get_provider(**common, framing="pad", views=1)
    cropped = manager.get_provider(**common, framing="crop", views=1)
    multi_view = manager.get_provider(**common, framing="pad", views=3)

    assert padded is not cropped
    assert padded is not multi_view
    assert padded is manager.get_provider(**common, framing="pad", views=1)


def test_framing_is_stored_with_the_collection(tmp_path) -> None:
    registry = CollectionMetadataService(tmp_path / "collections.json")

    registry.save(
        collection_name="products",
        model=CollectionModelConfig(
            provider="openclip",
            name="ViT-B-32",
            pretrained="p",
            vector_size=512,
            framing="pad",
            views=3,
        ),
    )

    stored = registry.get("products")
    assert stored.framing == "pad"
    assert stored.views == 3


def test_a_collection_written_before_framing_existed_keeps_centre_cropping(
    tmp_path,
) -> None:
    """Its vectors were built by cropping, so it must keep being searched that way."""
    path = tmp_path / "collections.json"
    path.write_text(
        '{"legacy": {"collection_name": "legacy", "embedding_provider": "openclip",'
        ' "embedding_model": "ViT-B-32", "embedding_pretrained": "p",'
        ' "vector_size": 512, "distance": "cosine", "created_at": "2026-01-01"}}'
    )

    stored = CollectionMetadataService(path).get("legacy")

    assert stored.framing == "crop"
    assert stored.views == 1


def test_default_framing_is_padding() -> None:
    assert Settings().image_framing == "pad"


@pytest.mark.parametrize("framing", ["squish", "stretch", ""])
def test_unknown_framing_is_refused(framing) -> None:
    with pytest.raises(ValidationError):
        Settings(image_framing=framing)
