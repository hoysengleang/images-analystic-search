"""Checks the ONNX provider against the real onnxruntime and tokenizers APIs.

The models here are tiny synthetic graphs rather than a real CLIP export, so
nothing is downloaded and the suite stays fast. They are still enough to catch
what actually breaks: a changed session-run signature, preprocessing that no
longer produces the tensor the model declares, tokenization that drops the
end-of-text marker, or a vector width that no longer matches the configuration.

Numerical agreement with PyTorch is not something a synthetic graph can prove.
`scripts/export_onnx.py` checks that at export time, against the real model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.errors import BadRequestError, ServiceUnavailableError
from app.embedding.providers.onnx_provider import CONTEXT_LENGTH, ONNXProvider

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

INPUT_SIZE = 224
VECTOR_SIZE = 512
OPSET_VERSION = 17
#: Kept below the installed onnxruntime's ceiling; raising it gains nothing.
IR_VERSION = 9


def build_image_model(path: Path) -> None:
    """A graph shaped like an image encoder: (N,3,S,S) float -> (N,512) float.

    Mean pooling the spatial axes means a differently coloured or differently
    framed image really does produce a different vector, which is what the
    framing and multi-view tests below rely on.
    """
    weights = _fixed_weights(3, VECTOR_SIZE)
    nodes = [
        onnx.helper.make_node(
            "ReduceMean", ["pixel_values"], ["pooled"], axes=[2, 3], keepdims=0
        ),
        onnx.helper.make_node("MatMul", ["pooled", "weights"], ["image_features"]),
    ]
    _write_model(
        path,
        nodes=nodes,
        initializer=weights,
        input=onnx.helper.make_tensor_value_info(
            "pixel_values",
            onnx.TensorProto.FLOAT,
            ["batch", 3, INPUT_SIZE, INPUT_SIZE],
        ),
        output=onnx.helper.make_tensor_value_info(
            "image_features", onnx.TensorProto.FLOAT, ["batch", VECTOR_SIZE]
        ),
    )


def build_text_model(path: Path, *, token_type=None) -> None:
    """A graph shaped like a text encoder: (N,77) int -> (N,512) float.

    int32 by default, matching what `scripts/export_onnx.py` produces: ONNX
    Runtime has no int64 ArgMax on CPU, and CLIP's text encoder argmaxes the
    token ids, so an int64 export fails to load at all.
    """
    token_type = onnx.TensorProto.INT32 if token_type is None else token_type
    weights = _fixed_weights(CONTEXT_LENGTH, VECTOR_SIZE)
    nodes = [
        onnx.helper.make_node(
            "Cast", ["input_ids"], ["as_float"], to=onnx.TensorProto.FLOAT
        ),
        onnx.helper.make_node("MatMul", ["as_float", "weights"], ["text_features"]),
    ]
    _write_model(
        path,
        nodes=nodes,
        initializer=weights,
        input=onnx.helper.make_tensor_value_info(
            "input_ids", token_type, ["batch", CONTEXT_LENGTH]
        ),
        output=onnx.helper.make_tensor_value_info(
            "text_features", onnx.TensorProto.FLOAT, ["batch", VECTOR_SIZE]
        ),
    )


def _fixed_weights(rows: int, columns: int):
    values = np.random.default_rng(seed=0).normal(size=(rows, columns))
    return onnx.helper.make_tensor(
        "weights", onnx.TensorProto.FLOAT, (rows, columns), values.astype(np.float32)
    )


def _write_model(path: Path, *, nodes, initializer, input, output) -> None:
    graph = onnx.helper.make_graph(
        nodes, path.stem, [input], [output], initializer=[initializer]
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", OPSET_VERSION)]
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())


@pytest.fixture(scope="module")
def model_files(tmp_path_factory, clip_tokenizer_file) -> dict:
    directory = tmp_path_factory.mktemp("onnx-models")
    paths = {
        "image_model_path": directory / "image_encoder.onnx",
        "text_model_path": directory / "text_encoder.onnx",
        "tokenizer_path": clip_tokenizer_file,
    }
    build_image_model(paths["image_model_path"])
    build_text_model(paths["text_model_path"])
    return {name: str(path) for name, path in paths.items()}


def make_provider(model_files: dict, **overrides) -> ONNXProvider:
    options = {
        "model_name": "ViT-B-32",
        "model_pretrained": "none",
        "vector_size": VECTOR_SIZE,
        **model_files,
        **overrides,
    }
    return ONNXProvider(**options)


@pytest.fixture(scope="module")
def provider(model_files: dict) -> ONNXProvider:
    return make_provider(model_files)


def is_unit_length(vector) -> bool:
    return sum(value * value for value in vector) == pytest.approx(1.0, abs=1e-4)


def test_text_support_follows_the_configured_encoder(model_files: dict) -> None:
    """The capability is per instance here, so /models must report the truth."""
    with_text = make_provider(model_files)
    image_only = make_provider(model_files, text_model_path=None, tokenizer_path=None)

    assert with_text.supports_text is True
    assert with_text.metadata.supports_text is True
    assert image_only.supports_text is False
    assert image_only.metadata.supports_text is False


def test_images_encode_to_normalized_vectors(provider: ONNXProvider) -> None:
    vectors = provider.embed_images(
        [
            Image.new("RGB", (64, 64), (255, 0, 0)),
            Image.new("RGBA", (64, 64), (0, 0, 255, 128)),
        ]
    )

    assert len(vectors) == 2
    assert all(len(vector) == VECTOR_SIZE for vector in vectors)
    assert all(is_unit_length(vector) for vector in vectors)


def test_text_encodes_into_the_same_vector_space(provider: ONNXProvider) -> None:
    text_vectors = provider.embed_texts(["red running shoe", "leather bag"])
    image_vector = provider.embed_image(Image.new("RGB", (64, 64), (255, 0, 0)))

    assert len(text_vectors) == 2
    # Same width as the image vectors, which is what makes them comparable.
    assert all(len(vector) == VECTOR_SIZE for vector in text_vectors)
    assert len(image_vector) == VECTOR_SIZE
    assert all(is_unit_length(vector) for vector in text_vectors)


def test_empty_batches_short_circuit(provider: ONNXProvider) -> None:
    assert provider.embed_images([]) == []
    assert provider.embed_texts([]) == []


def test_input_size_is_read_from_the_model(provider: ONNXProvider) -> None:
    """Preprocessing has to match the export, not a compiled-in constant."""
    provider.warmup()

    assert provider._input_size == INPUT_SIZE


def test_padding_and_cropping_produce_different_vectors(model_files: dict) -> None:
    """If they matched, framing would not need pinning per collection."""
    tall = Image.new("RGB", (200, 500), (250, 250, 250))
    tall.paste(Image.new("RGB", (200, 100), (220, 30, 30)), (0, 0))
    tall.paste(Image.new("RGB", (200, 100), (30, 30, 220)), (0, 400))

    cropped = make_provider(model_files, framing="crop").embed_image(tall)
    padded = make_provider(model_files, framing="pad").embed_image(tall)

    similarity = sum(a * b for a, b in zip(cropped, padded))
    assert similarity < 0.999, "framing must change what the model sees"


def test_multiple_views_still_produce_one_unit_vector(model_files: dict) -> None:
    provider = make_provider(model_files, framing="pad", views=3)

    vectors = provider.embed_images(
        [
            Image.new("RGB", (300, 500), (200, 60, 60)),
            Image.new("RGB", (500, 300), (60, 200, 60)),
        ]
    )

    assert len(vectors) == 2
    assert all(len(vector) == VECTOR_SIZE for vector in vectors)
    assert all(is_unit_length(vector) for vector in vectors)


def test_text_longer_than_the_context_window_keeps_its_end_marker(
    provider: ONNXProvider,
) -> None:
    """The encoder reads its output at the end-of-text position."""
    provider.warmup()
    tokens = provider._tokenize([" ".join(["word"] * (CONTEXT_LENGTH * 2))])

    assert tokens.shape == (1, CONTEXT_LENGTH)
    assert tokens[0][-1] == provider._end_of_text_id


def test_short_text_is_padded_with_zeros(provider: ONNXProvider) -> None:
    """open_clip pads with zeros, so a matching export expects zeros."""
    provider.warmup()
    tokens = provider._tokenize(["red shoe"])

    assert tokens.shape == (1, CONTEXT_LENGTH)
    assert tokens[0][-1] == 0


def test_text_search_is_refused_without_a_text_encoder(model_files: dict) -> None:
    image_only = make_provider(model_files, text_model_path=None, tokenizer_path=None)

    with pytest.raises(BadRequestError) as exc_info:
        image_only.embed_text("red running shoe")

    assert exc_info.value.code == "TEXT_SEARCH_NOT_SUPPORTED"


def test_vector_size_mismatch_is_reported(model_files: dict) -> None:
    misconfigured = make_provider(model_files, vector_size=64)  # really 512

    with pytest.raises(ServiceUnavailableError) as exc_info:
        misconfigured.embed_text("red running shoe")

    assert exc_info.value.code == "ONNX_VECTOR_SIZE_MISMATCH"


def test_a_missing_model_file_names_the_path(model_files: dict) -> None:
    provider = make_provider(model_files, image_model_path="/no/such/model.onnx")

    with pytest.raises(ServiceUnavailableError) as exc_info:
        provider.embed_image(Image.new("RGB", (64, 64), (255, 0, 0)))

    assert exc_info.value.code == "ONNX_MODEL_LOAD_FAILED"
    assert exc_info.value.details["path"] == "/no/such/model.onnx"


def test_the_image_encoder_is_required(model_files: dict) -> None:
    """The manager drops unset options, so this must be a readable message."""
    with pytest.raises(ValueError, match="ONNX_IMAGE_MODEL_PATH"):
        make_provider(model_files, image_model_path=None)


def test_a_text_encoder_without_a_tokenizer_is_refused(model_files: dict) -> None:
    with pytest.raises(ValueError, match="ONNX_TOKENIZER_PATH"):
        make_provider(model_files, tokenizer_path=None)


@pytest.mark.parametrize("views", (1, 2, 3))
@pytest.mark.parametrize("framing", ("crop", "pad"))
def test_preprocessing_matches_the_torchvision_transform(
    model_files: dict, framing: str, views: int
) -> None:
    """Everything this provider claims rests on producing the same tensor.

    The ONNX path cannot use the torchvision transform OpenCLIP hands back, so
    it reimplements it on Pillow. Two details are easy to get wrong and cost
    nothing visible when you do: torchvision truncates the scaled long edge but
    rounds the crop offset, and under "crop" framing the extra views come from
    the full-resolution photo rather than from an already-framed square. Either
    mistake shifts what the model sees and degrades search while every other
    test still passes.
    """
    torch = pytest.importorskip("torch")
    open_clip = pytest.importorskip("open_clip")

    from app.embedding.providers.openclip_provider import OpenCLIPProvider

    preprocess = open_clip.image_transform(INPUT_SIZE, is_train=False)

    exported = make_provider(model_files, framing=framing, views=views)
    exported.warmup()  # reads the input size off the model, as in production

    reference = OpenCLIPProvider(
        model_name="ViT-B-32",
        model_pretrained="none",
        vector_size=VECTOR_SIZE,
        framing=framing,
        views=views,
    )
    reference._preprocess = preprocess
    reference._input_size = INPUT_SIZE

    rng = np.random.default_rng(seed=11)
    # Portrait, landscape, square, and two extreme aspect ratios: the rounding
    # differences show up on some of these shapes and not on others.
    for width, height in ((640, 480), (300, 900), (224, 224), (1000, 137), (17, 4000)):
        image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))

        expected = torch.stack(
            [preprocess(view) for view in reference._views_of(image)]
        ).numpy()
        actual = np.stack(
            [exported._to_input_array(view) for view in exported._views_of(image)]
        )

        assert actual.shape == expected.shape
        assert np.array_equal(actual, expected), (
            f"{width}x{height} preprocessing drifted from the reference transform"
        )


def test_token_ids_use_the_width_the_encoder_declares(
    provider: ONNXProvider, tmp_path: Path
) -> None:
    """A mismatch here is not a wrong answer, it is a session that will not load.

    ONNX Runtime's CPU provider has no int64 ArgMax, and CLIP's text encoder
    argmaxes the token ids to find end-of-text, so real exports are int32. An
    int64 export stays readable wherever it does run.
    """
    provider.warmup()
    assert provider._tokenize(["red shoe"]).dtype == np.int32

    wide = tmp_path / "text_encoder_int64.onnx"
    build_text_model(wide, token_type=onnx.TensorProto.INT64)
    assert ONNXProvider._token_dtype_of(_session(wide)) == np.int64


def _session(path: Path):
    import onnxruntime

    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
