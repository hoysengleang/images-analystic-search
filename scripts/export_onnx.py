"""Export an OpenCLIP checkpoint to the ONNX files the `onnx` provider serves.

Run this once, on any machine that can install PyTorch. The output is three
small files you copy to the server, which then needs only onnxruntime — tens
of megabytes instead of the better part of a gigabyte.

    PYTHONPATH=. python scripts/export_onnx.py --output-dir data/models

The export is verified before it is accepted: the exported files are loaded
back through the real :class:`ONNXProvider` and compared against the PyTorch
provider on the same inputs. A mismatch fails the run, so an export that would
have quietly degraded search never reaches a server.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from app.embedding.providers.onnx_provider import CONTEXT_LENGTH, ONNXProvider
from app.embedding.providers.openclip_provider import OpenCLIPProvider

#: Every standard CLIP checkpoint shares this 49408-token BPE vocabulary, so
#: one tokenizer file serves every model this script can export.
TOKENIZER_REPO = "openai/clip-vit-base-patch32"
TOKENIZER_FILE = "tokenizer.json"

IMAGE_ENCODER_FILE = "image_encoder.onnx"
TEXT_ENCODER_FILE = "text_encoder.onnx"

#: 17 is old enough for any onnxruntime we support and new enough for ViT.
OPSET_VERSION = 17

#: Cosine similarity below this means the export changed what the model sees.
PARITY_THRESHOLD = 0.999

VERIFICATION_IMAGES = (
    ("RGB", (640, 480), (200, 60, 60)),
    ("RGB", (300, 900), (60, 200, 90)),
    ("RGB", (224, 224), (40, 70, 210)),
)
VERIFICATION_TEXTS = (
    "a red running shoe",
    "brown leather handbag with a gold clasp",
    "minimalist white ceramic mug",
)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    model, tokenizer = load_openclip(args.model, args.pretrained)
    input_size = visual_input_size(model)

    image_path = output_dir / IMAGE_ENCODER_FILE
    text_path = output_dir / TEXT_ENCODER_FILE
    tokenizer_path = output_dir / TOKENIZER_FILE

    print(f"Exporting {args.model} / {args.pretrained} at {input_size}px")

    export_encoder(
        module=encoder_module(model, "encode_image"),
        example=torch.randn(1, 3, input_size, input_size),
        path=image_path,
        input_name="pixel_values",
        output_name="image_features",
    )
    export_encoder(
        module=encoder_module(model, "encode_text"),
        # int32, not int64: ONNX Runtime has no int64 ArgMax on CPU, and
        # encode_text argmaxes the ids to find the end-of-text position.
        example=tokenizer(["a photo"]).to(torch.int32),
        path=text_path,
        input_name="input_ids",
        output_name="text_features",
    )
    fetch_tokenizer(tokenizer_path)

    vector_size = output_vector_size(model, input_size)
    verify(
        model_name=args.model,
        pretrained=args.pretrained,
        vector_size=vector_size,
        framing=args.framing,
        image_path=image_path,
        text_path=text_path,
        tokenizer_path=tokenizer_path,
    )

    report(output_dir, image_path, text_path, tokenizer_path, vector_size)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ViT-B-32", help="OpenCLIP architecture")
    parser.add_argument(
        "--pretrained",
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained tag",
    )
    parser.add_argument(
        "--output-dir",
        default="data/models",
        help="Directory the three exported files are written to",
    )
    parser.add_argument(
        "--framing",
        default="pad",
        choices=("pad", "crop"),
        help="Framing to verify parity under; use the one you will run with",
    )
    return parser.parse_args()


def encoder_module(model, method_name: str):
    """Wrap one encode method so the exporter sees a plain forward pass."""
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, inputs):
            return getattr(self.model, method_name)(inputs)

    return Encoder().eval()


def load_openclip(model_name: str, pretrained: str):
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device="cpu",
    )
    model.eval()
    return model, open_clip.get_tokenizer(model_name)


def visual_input_size(model) -> int:
    size = getattr(model.visual, "image_size", 224)
    return int(size[0] if isinstance(size, (tuple, list)) else size)


def output_vector_size(model, input_size: int) -> int:
    """Ask the model how wide its vectors really are, rather than assume."""
    import torch

    with torch.inference_mode():
        features = model.encode_image(torch.zeros(1, 3, input_size, input_size))
    return int(features.shape[-1])


@contextmanager
def fused_attention_disabled():
    """Turn PyTorch's fused attention kernel off for the duration of an export.

    An eval-mode ``nn.MultiheadAttention`` dispatches to the fused
    ``_native_multi_head_attention``, which has no ONNX translation, so the
    export fails outright rather than degrading. The decomposed path computes
    the same thing and does export; the parity check afterwards is what holds
    that claim to account.
    """
    import torch

    backend = getattr(torch.backends, "mha", None)
    if backend is None:  # torch < 2.2 has no switch, and no fast path to avoid
        yield
        return

    previous = backend.get_fastpath_enabled()
    backend.set_fastpath_enabled(False)
    try:
        yield
    finally:
        backend.set_fastpath_enabled(previous)


def export_encoder(
    *, module, example, path: Path, input_name: str, output_name: str
) -> None:
    """Write one encoder to ONNX with a dynamic batch axis."""
    import torch

    # PyTorch 2.9 makes the dynamo exporter the default. Pin the TorchScript
    # one: it is the path this script's output has been verified against, and a
    # silent switch would change the graph under an unchanged command. Checked
    # by signature rather than by catching TypeError, which would also swallow
    # a real error from inside the export.
    options = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False

    with fused_attention_disabled(), torch.inference_mode():
        torch.onnx.export(
            module,
            (example,),
            str(path),
            input_names=[input_name],
            output_names=[output_name],
            # Only the batch axis moves. Height and width stay pinned so the
            # provider can read the input size back off the model.
            dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
            opset_version=OPSET_VERSION,
            **options,
        )
    print(f"  wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def fetch_tokenizer(path: Path) -> None:
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=TOKENIZER_REPO, filename=TOKENIZER_FILE)
    path.write_bytes(Path(downloaded).read_bytes())
    print(f"  wrote {path.name} ({path.stat().st_size / 1e3:.0f} KB)")


def verify(
    *,
    model_name: str,
    pretrained: str,
    vector_size: int,
    framing: str,
    image_path: Path,
    text_path: Path,
    tokenizer_path: Path,
) -> None:
    """Fail the export unless ONNX reproduces PyTorch on real inputs."""
    reference = OpenCLIPProvider(
        model_name=model_name,
        model_pretrained=pretrained,
        vector_size=vector_size,
        framing=framing,
    )
    exported = ONNXProvider(
        model_name=model_name,
        model_pretrained=pretrained,
        vector_size=vector_size,
        image_model_path=str(image_path),
        text_model_path=str(text_path),
        tokenizer_path=str(tokenizer_path),
        framing=framing,
    )

    images = [Image.new(mode, size, colour) for mode, size, colour in VERIFICATION_IMAGES]
    try:
        check_parity(
            "image",
            reference.embed_images(images),
            exported.embed_images(images),
        )
    finally:
        for image in images:
            image.close()

    texts = list(VERIFICATION_TEXTS)
    check_parity("text", reference.embed_texts(texts), exported.embed_texts(texts))

    long_text = " ".join(["word"] * (CONTEXT_LENGTH * 2))
    check_parity(
        "truncated text",
        reference.embed_texts([long_text]),
        exported.embed_texts([long_text]),
    )


def check_parity(label: str, reference: list, exported: list) -> None:
    similarities = [
        float(np.dot(left, right)) for left, right in zip(reference, exported)
    ]
    worst = min(similarities)
    if worst < PARITY_THRESHOLD:
        raise SystemExit(
            f"{label} parity check failed: cosine similarity to PyTorch was "
            f"{worst:.5f}, below {PARITY_THRESHOLD}. The exported model does "
            f"not match the original; do not deploy these files."
        )
    print(f"  {label} parity OK (worst cosine {worst:.5f})")


def report(
    output_dir: Path,
    image_path: Path,
    text_path: Path,
    tokenizer_path: Path,
    vector_size: int,
) -> None:
    total_mb = sum(p.stat().st_size for p in (image_path, text_path)) / 1e6
    print(f"\nExported {total_mb:.1f} MB of model to {output_dir}/\n")
    print("Point the service at them and drop torch from the image:\n")
    print("  DEFAULT_EMBEDDING_PROVIDER=onnx")
    print(f"  DEFAULT_VECTOR_SIZE={vector_size}")
    print(f"  ONNX_IMAGE_MODEL_PATH={image_path}")
    print(f"  ONNX_TEXT_MODEL_PATH={text_path}")
    print(f"  ONNX_TOKENIZER_PATH={tokenizer_path}")
    print("\n  pip install -r requirements-onnx.txt")


if __name__ == "__main__":
    sys.exit(main())
