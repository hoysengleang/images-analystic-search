"""Checks the OWL-ViT detector against the real onnxruntime API.

The model is a synthetic graph that ignores its inputs and emits fixed logits
and boxes, which is what makes the assertions below possible: post-processing is
the part that silently goes wrong, and pinning it needs known outputs rather
than a real 600 MB detector. Whether the exported weights themselves match
PyTorch is checked at export time by `scripts/export_detector_onnx.py`.

What is genuinely under test here: centre-form box decoding, mapping boxes back
off the padding OWL-ViT adds to square an image, the score threshold,
non-maximum suppression, the region cap, and prompt labelling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.errors import BadRequestError, ServiceUnavailableError
from app.detection.providers.owlvit_onnx_provider import OWLViTONNXProvider

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

INPUT_SIZE = 96
CONTEXT_LENGTH = 16
OPSET_VERSION = 17
IR_VERSION = 9

PROMPTS = ("a handbag", "a shoe")


def build_detector_model(path: Path, *, logits: np.ndarray, boxes: np.ndarray) -> None:
    """A graph that declares OWL-ViT's inputs and returns fixed predictions.

    The inputs are declared but unused: they exist so the provider's feed
    building and shape probing run against a real session, while the outputs
    stay under the test's control.
    """
    initializers = [
        onnx.helper.make_tensor(
            "logits_value",
            onnx.TensorProto.FLOAT,
            logits.shape,
            logits.astype(np.float32).ravel(),
        ),
        onnx.helper.make_tensor(
            "boxes_value",
            onnx.TensorProto.FLOAT,
            boxes.shape,
            boxes.astype(np.float32).ravel(),
        ),
    ]
    nodes = [
        onnx.helper.make_node("Identity", ["logits_value"], ["logits"]),
        onnx.helper.make_node("Identity", ["boxes_value"], ["pred_boxes"]),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "owlvit-stub",
        [
            onnx.helper.make_tensor_value_info(
                "pixel_values",
                onnx.TensorProto.FLOAT,
                ["batch", 3, INPUT_SIZE, INPUT_SIZE],
            ),
            onnx.helper.make_tensor_value_info(
                "input_ids",
                onnx.TensorProto.INT32,
                ["queries", CONTEXT_LENGTH],
            ),
        ],
        [
            onnx.helper.make_tensor_value_info(
                "logits", onnx.TensorProto.FLOAT, list(logits.shape)
            ),
            onnx.helper.make_tensor_value_info(
                "pred_boxes", onnx.TensorProto.FLOAT, list(boxes.shape)
            ),
        ],
        initializer=initializers,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", OPSET_VERSION)]
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())


def logit(probability: float) -> float:
    """The raw score that sigmoids to `probability`, so tests read in probabilities."""
    return float(np.log(probability / (1.0 - probability)))


def make_provider(
    tmp_path: Path,
    clip_tokenizer_file: Path,
    *,
    scores: list,
    boxes: list,
) -> OWLViTONNXProvider:
    """Build a detector whose model predicts exactly `scores` and `boxes`.

    `scores` is one probability per prompt per candidate box.
    """
    logits = np.array([[[logit(value) for value in row] for row in scores]])
    model_path = tmp_path / "detector.onnx"
    build_detector_model(
        model_path,
        logits=logits,
        boxes=np.array([boxes], dtype=np.float32),
    )
    return OWLViTONNXProvider(
        model_path=str(model_path),
        tokenizer_path=str(clip_tokenizer_file),
    )


def detect(provider: OWLViTONNXProvider, image_size=(96, 96), **overrides) -> list:
    options = {"prompts": PROMPTS, "max_regions": 5, "min_score": 0.1, **overrides}
    image = Image.new("RGB", image_size, (120, 120, 120))
    try:
        return provider.detect(image, **options)
    finally:
        image.close()


def test_the_input_size_is_read_from_the_model(tmp_path, clip_tokenizer_file) -> None:
    """Preprocessing must match the export, not a compiled-in constant."""
    provider = make_provider(
        tmp_path, clip_tokenizer_file, scores=[[0.9, 0.1]], boxes=[[0.5, 0.5, 0.4, 0.4]]
    )

    provider.warmup()

    assert provider._input_size == INPUT_SIZE
    assert provider._context_length == CONTEXT_LENGTH


def test_boxes_are_decoded_from_centre_form(tmp_path, clip_tokenizer_file) -> None:
    """OWL-ViT predicts centre-x, centre-y, width, height; the API takes corners."""
    provider = make_provider(
        tmp_path, clip_tokenizer_file, scores=[[0.9, 0.1]], boxes=[[0.5, 0.5, 0.4, 0.6]]
    )

    region = detect(provider)[0]

    assert region.box.x == pytest.approx(0.3)
    assert region.box.y == pytest.approx(0.2)
    assert region.box.width == pytest.approx(0.4)
    assert region.box.height == pytest.approx(0.6)


@pytest.mark.parametrize("image_size", [(200, 100), (100, 200), (96, 96), (1920, 1080)])
def test_boxes_are_independent_of_the_image_shape(
    tmp_path, clip_tokenizer_file, image_size
) -> None:
    """OWL-ViT stretches an image to a square rather than padding or cropping.

    Scaling each axis on its own leaves normalized coordinates untouched, so the
    same prediction describes the same part of the picture whatever its aspect
    ratio. If this ever starts depending on the shape, boxes are being mapped
    back from a square the model never saw.
    """
    provider = make_provider(
        tmp_path, clip_tokenizer_file, scores=[[0.9, 0.1]], boxes=[[0.5, 0.25, 0.4, 0.2]]
    )

    region = detect(provider, image_size=image_size)[0]

    assert region.box.x == pytest.approx(0.3)
    assert region.box.y == pytest.approx(0.15)
    assert region.box.width == pytest.approx(0.4)
    assert region.box.height == pytest.approx(0.2)


def test_low_confidence_regions_are_dropped(tmp_path, clip_tokenizer_file) -> None:
    provider = make_provider(
        tmp_path,
        clip_tokenizer_file,
        scores=[[0.90, 0.05], [0.20, 0.10]],
        boxes=[[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]],
    )

    assert len(detect(provider, min_score=0.1)) == 2
    assert len(detect(provider, min_score=0.5)) == 1
    assert detect(provider, min_score=0.95) == []


def test_overlapping_regions_are_suppressed(tmp_path, clip_tokenizer_file) -> None:
    """Two boxes on the same object should not become two results."""
    provider = make_provider(
        tmp_path,
        clip_tokenizer_file,
        scores=[[0.90, 0.05], [0.85, 0.05], [0.80, 0.05]],
        boxes=[
            [0.30, 0.30, 0.40, 0.40],
            [0.31, 0.31, 0.40, 0.40],  # essentially the same box
            [0.80, 0.80, 0.20, 0.20],  # a different object
        ],
    )

    regions = detect(provider)

    assert len(regions) == 2
    # The stronger of the overlapping pair is the one that survives.
    assert regions[0].score == pytest.approx(0.90, abs=1e-4)


def test_the_region_cap_is_honoured(tmp_path, clip_tokenizer_file) -> None:
    provider = make_provider(
        tmp_path,
        clip_tokenizer_file,
        scores=[[0.9, 0.1], [0.8, 0.1], [0.7, 0.1], [0.6, 0.1]],
        boxes=[
            [0.1, 0.1, 0.1, 0.1],
            [0.4, 0.1, 0.1, 0.1],
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
        ],
    )

    assert len(detect(provider, max_regions=2)) == 2
    assert detect(provider, max_regions=0) == []


def test_a_region_is_labelled_with_its_strongest_prompt(
    tmp_path, clip_tokenizer_file
) -> None:
    provider = make_provider(
        tmp_path,
        clip_tokenizer_file,
        scores=[[0.20, 0.90], [0.90, 0.20]],
        boxes=[[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]],
    )

    labels = [region.label for region in detect(provider)]

    assert labels == ["a shoe", "a handbag"]


def test_detection_needs_something_to_look_for(tmp_path, clip_tokenizer_file) -> None:
    """Open-vocabulary means no class list, so prompts are not optional."""
    provider = make_provider(
        tmp_path, clip_tokenizer_file, scores=[[0.9, 0.1]], boxes=[[0.5, 0.5, 0.2, 0.2]]
    )

    with pytest.raises(BadRequestError) as exc_info:
        detect(provider, prompts=[])

    assert exc_info.value.code == "DETECTOR_PROMPTS_REQUIRED"


def test_prompts_are_padded_and_masked_to_the_context_length(
    tmp_path, clip_tokenizer_file
) -> None:
    """The mask, not the padding value, is what makes the model ignore padding."""
    provider = make_provider(
        tmp_path, clip_tokenizer_file, scores=[[0.9, 0.1]], boxes=[[0.5, 0.5, 0.2, 0.2]]
    )
    provider.warmup()

    tokens, mask = provider._tokenize(["red bag", "shoe"])

    assert tokens.shape == (2, CONTEXT_LENGTH)
    assert mask.shape == (2, CONTEXT_LENGTH)
    # "red bag" is start + 2 words + end; "shoe" is start + 1 word + end.
    assert mask[0].sum() == 4
    assert mask[1].sum() == 3
    assert tokens[0][-1] == provider._end_of_text_id


def test_a_missing_model_file_names_the_path(tmp_path, clip_tokenizer_file) -> None:
    provider = OWLViTONNXProvider(
        model_path="/no/such/detector.onnx",
        tokenizer_path=str(clip_tokenizer_file),
    )

    with pytest.raises(ServiceUnavailableError) as exc_info:
        detect(provider)

    assert exc_info.value.code == "DETECTOR_MODEL_LOAD_FAILED"
    assert exc_info.value.details["path"] == "/no/such/detector.onnx"


def test_both_model_files_are_required() -> None:
    with pytest.raises(ValueError, match="DETECTOR_MODEL_PATH"):
        OWLViTONNXProvider(model_path=None, tokenizer_path="/tmp/tokenizer.json")
