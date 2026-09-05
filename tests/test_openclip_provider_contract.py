"""Checks the OpenCLIP provider against the real open_clip and torch APIs.

The model architecture is built with random weights (`pretrained=None`), so
nothing is downloaded. That is enough to catch the breakages that matter here:
a changed tokenizer call, a renamed encode method, or a vector size that no
longer matches the configured one.

Marked `slow` because building the architecture takes a few seconds. Skip it
while iterating with:  python -m pytest -m "not slow"
"""

import pytest
from PIL import Image

from app.core.errors import ServiceUnavailableError
from app.embedding.providers.openclip_provider import OpenCLIPProvider

pytest.importorskip("torch")
pytest.importorskip("open_clip")

pytestmark = pytest.mark.slow

MODEL_NAME = "ViT-B-32"
VECTOR_SIZE = 512


class RandomWeightsProvider(OpenCLIPProvider):
    """Builds the real architecture without fetching pretrained weights."""

    def _load_model(self) -> None:
        import open_clip
        import torch

        device = torch.device("cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=None,
            device=device,
        )
        model.eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._device = device
        self._model = model


@pytest.fixture(scope="module")
def provider() -> OpenCLIPProvider:
    return RandomWeightsProvider(
        model_name=MODEL_NAME,
        model_pretrained="none",
        vector_size=VECTOR_SIZE,
    )


def is_unit_length(vector) -> bool:
    return sum(value * value for value in vector) == pytest.approx(1.0, abs=1e-4)


def test_provider_declares_text_support() -> None:
    assert OpenCLIPProvider.supports_text is True


def test_images_encode_to_normalized_vectors(provider: OpenCLIPProvider) -> None:
    vectors = provider.embed_images(
        [
            Image.new("RGB", (64, 64), (255, 0, 0)),
            Image.new("RGBA", (64, 64), (0, 0, 255, 128)),
        ]
    )

    assert len(vectors) == 2
    assert all(len(vector) == VECTOR_SIZE for vector in vectors)
    assert all(is_unit_length(vector) for vector in vectors)


def test_text_encodes_into_the_same_vector_space(provider: OpenCLIPProvider) -> None:
    text_vectors = provider.embed_texts(["red running shoe", "a leather bag"])
    image_vector = provider.embed_image(Image.new("RGB", (64, 64), (255, 0, 0)))

    assert len(text_vectors) == 2
    # Same width as the image vectors, which is what makes them comparable.
    assert all(len(vector) == VECTOR_SIZE for vector in text_vectors)
    assert len(image_vector) == VECTOR_SIZE
    assert all(is_unit_length(vector) for vector in text_vectors)


def test_empty_batches_short_circuit(provider: OpenCLIPProvider) -> None:
    assert provider.embed_images([]) == []
    assert provider.embed_texts([]) == []


def test_padding_and_cropping_produce_different_vectors() -> None:
    """If they matched, framing would not need pinning per collection."""
    tall = Image.new("RGB", (200, 500), (250, 250, 250))
    tall.paste(Image.new("RGB", (200, 100), (220, 30, 30)), (0, 0))
    tall.paste(Image.new("RGB", (200, 100), (30, 30, 220)), (0, 400))

    cropped = RandomWeightsProvider(
        model_name=MODEL_NAME,
        model_pretrained="none",
        vector_size=VECTOR_SIZE,
        framing="crop",
    ).embed_image(tall)
    padded = RandomWeightsProvider(
        model_name=MODEL_NAME,
        model_pretrained="none",
        vector_size=VECTOR_SIZE,
        framing="pad",
    ).embed_image(tall)

    similarity = sum(a * b for a, b in zip(cropped, padded))
    assert similarity < 0.999, "framing must change what the model sees"


def test_multiple_views_still_produce_one_unit_vector() -> None:
    provider = RandomWeightsProvider(
        model_name=MODEL_NAME,
        model_pretrained="none",
        vector_size=VECTOR_SIZE,
        framing="pad",
        views=3,
    )

    vectors = provider.embed_images(
        [
            Image.new("RGB", (300, 500), (200, 60, 60)),
            Image.new("RGB", (500, 300), (60, 200, 60)),
        ]
    )

    assert len(vectors) == 2
    assert all(len(vector) == VECTOR_SIZE for vector in vectors)
    assert all(is_unit_length(vector) for vector in vectors)


def test_vector_size_mismatch_is_reported(provider: OpenCLIPProvider) -> None:
    misconfigured = RandomWeightsProvider(
        model_name=MODEL_NAME,
        model_pretrained="none",
        vector_size=64,  # the model really produces 512
    )

    with pytest.raises(ServiceUnavailableError) as exc_info:
        misconfigured.embed_text("red running shoe")

    assert exc_info.value.code == "OPENCLIP_VECTOR_SIZE_MISMATCH"
