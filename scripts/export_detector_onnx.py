"""Export an OWL-ViT detector to the ONNX files the `owlvit` provider serves.

Run this once, on a machine that can install PyTorch and transformers. The
output is a model and a tokenizer you copy to the server, which then needs only
onnxruntime.

    PYTHONPATH=. python scripts/export_detector_onnx.py --output-dir data/models

The export is verified before it is accepted, in two halves that together cover
the whole pipeline:

* the tensor this repository builds from an image is compared against the one
  the reference processor builds, and
* the exported graph's outputs are compared against PyTorch's on those tensors.

Post-processing - decoding boxes, undoing the padding, suppression - is covered
by tests/test_owlvit_detector_contract.py, which needs no weights at all.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from app.detection.providers.owlvit_onnx_provider import OWLViTONNXProvider

DETECTOR_FILE = "detector.onnx"
TOKENIZER_FILE = "detector_tokenizer.json"

#: 17 is old enough for any onnxruntime we support and new enough for ViT.
OPSET_VERSION = 17

#: Absolute tolerance on the exported graph's outputs against PyTorch.
PARITY_TOLERANCE = 1e-3

#: Deliberately awkward shapes: squaring the image is where preprocessing goes
#: wrong, and a square test picture would never exercise it.
VERIFICATION_SIZES = ((640, 480), (300, 900), (512, 512))
VERIFICATION_PROMPTS = ("a handbag", "a shoe")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / DETECTOR_FILE
    tokenizer_path = output_dir / TOKENIZER_FILE

    print(f"Exporting {args.model}")
    model, processor = load_owlvit(args.model)

    export_detector(model, processor, model_path)
    fetch_tokenizer(args.model, tokenizer_path)

    provider = OWLViTONNXProvider(
        model_path=str(model_path),
        tokenizer_path=str(tokenizer_path),
    )
    provider.warmup()

    verify_preprocessing(provider, processor)
    verify_graph(provider, model, processor)

    report(model_path, tokenizer_path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="google/owlvit-base-patch32",
        help="Hugging Face model id for an OWL-ViT checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="data/models",
        help="Directory the exported files are written to",
    )
    return parser.parse_args()


def load_owlvit(model_id: str):
    from transformers import AutoProcessor, OwlViTForObjectDetection

    # The fused attention kernels have no ONNX translation, so ask for the
    # plain implementation up front rather than failing deep inside the export.
    model = OwlViTForObjectDetection.from_pretrained(
        model_id,
        attn_implementation="eager",
    )
    model.eval()
    return model, AutoProcessor.from_pretrained(model_id)


@contextmanager
def fused_attention_disabled():
    """Turn PyTorch's fused attention kernel off for the duration of an export."""
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


def export_detector(model, processor, path: Path) -> None:
    import torch

    class Detector(torch.nn.Module):
        """Returns bare tensors: the exporter cannot see into an output object."""

        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values, input_ids, attention_mask):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )
            return outputs.logits, outputs.pred_boxes

    example = processor(
        text=[list(VERIFICATION_PROMPTS)],
        images=Image.new("RGB", (640, 480)),
        return_tensors="pt",
    )

    # PyTorch 2.9 makes the dynamo exporter the default. Pin the TorchScript
    # one: it is the path this script's output is verified against.
    options = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False

    with fused_attention_disabled(), torch.inference_mode():
        torch.onnx.export(
            Detector().eval(),
            (
                example["pixel_values"],
                example["input_ids"].to(torch.int32),
                example["attention_mask"].to(torch.int32),
            ),
            str(path),
            input_names=["pixel_values", "input_ids", "attention_mask"],
            output_names=["logits", "pred_boxes"],
            # The image size stays pinned so the provider can read it back; only
            # the batch and the number of prompts vary.
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "input_ids": {0: "queries"},
                "attention_mask": {0: "queries"},
                "logits": {0: "batch", 2: "queries"},
                "pred_boxes": {0: "batch"},
            },
            opset_version=OPSET_VERSION,
            **options,
        )

    print(f"  wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def fetch_tokenizer(model_id: str, path: Path) -> None:
    """Write the model's own tokenizer in the format the runtime can load.

    Converted from the checkpoint rather than downloaded: most OWL-ViT repos
    ship the slow tokenizer's vocabulary and merges instead of a `tokenizer.json`,
    and converting guarantees the file matches the model being exported.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.backend_tokenizer.save(str(path))
    print(f"  wrote {path.name} ({path.stat().st_size / 1e3:.0f} KB)")


def verify_preprocessing(provider: OWLViTONNXProvider, processor) -> None:
    """The tensor we build must be the tensor the model was trained to expect."""
    worst = 0.0
    for width, height in VERIFICATION_SIZES:
        image = random_image(width, height)
        try:
            reference = processor(images=image, return_tensors="np")["pixel_values"]
            ours = provider._to_input_array(image)
        finally:
            image.close()

        if ours.shape != reference.shape:
            raise SystemExit(
                f"preprocessing shape mismatch at {width}x{height}: "
                f"{ours.shape} against {reference.shape}"
            )
        worst = max(worst, float(np.abs(ours - reference).max()))

    if worst > PARITY_TOLERANCE:
        raise SystemExit(
            f"preprocessing differs from the reference processor by {worst:.5f}. "
            f"The detector would see a different picture; do not deploy this."
        )
    print(f"  preprocessing parity OK (worst difference {worst:.6f})")


def verify_graph(provider: OWLViTONNXProvider, model, processor) -> None:
    """The exported graph must compute what PyTorch computes."""
    import torch

    image = random_image(640, 480)
    try:
        inputs = processor(
            text=[list(VERIFICATION_PROMPTS)],
            images=image,
            return_tensors="pt",
        )
    finally:
        image.close()

    with torch.inference_mode():
        expected = model(**inputs)

    logits, boxes = provider._run(
        inputs["pixel_values"].numpy(),
        inputs["input_ids"].numpy().astype(provider._token_dtype),
        inputs["attention_mask"].numpy().astype(provider._token_dtype),
    )

    for name, ours, reference in (
        ("logits", logits, expected.logits.numpy()),
        ("boxes", boxes, expected.pred_boxes.numpy()),
    ):
        difference = float(np.abs(ours - reference).max())
        if difference > PARITY_TOLERANCE:
            raise SystemExit(
                f"{name} parity check failed: the exported graph differs from "
                f"PyTorch by {difference:.5f}. Do not deploy these files."
            )
        print(f"  {name} parity OK (worst difference {difference:.6f})")


def random_image(width: int, height: int) -> Image.Image:
    pixels = np.random.default_rng(seed=13).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    return Image.fromarray(pixels)


def report(model_path: Path, tokenizer_path: Path) -> None:
    print(f"\nExported {model_path.stat().st_size / 1e6:.1f} MB\n")
    print("Turn on query-side detection with:\n")
    print("  DETECTOR_PROVIDER=owlvit")
    print(f"  DETECTOR_MODEL_PATH={model_path}")
    print(f"  DETECTOR_TOKENIZER_PATH={tokenizer_path}")
    print('  DETECTOR_PROMPTS=["a product"]')
    print("\n  pip install -r requirements-onnx.txt")


if __name__ == "__main__":
    sys.exit(main())
