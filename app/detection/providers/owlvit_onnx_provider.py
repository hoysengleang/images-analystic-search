"""OWL-ViT object detection through ONNX Runtime.

OWL-ViT is open-vocabulary: it has no fixed class list and instead scores each
candidate box against text prompts you supply, so it finds a necklace or a
cushion as readily as the handful of categories a COCO-trained detector knows.
That matters for a product catalogue, where the class list is whatever the
merchant sells.

Running it through ONNX Runtime rather than PyTorch keeps the detector optional
without dragging torch back into the image. Build the model files with
``scripts/export_detector_onnx.py``.

This is OWL-ViT v1 rather than OWLv2. OWLv2 is the stronger model, but its
reference preprocessing is a scikit-image resize - a Gaussian anti-alias then a
scipy zoom - which cannot be reproduced with Pillow, so supporting it would mean
depending on scipy and could not be checked for exact parity. v1 stretches the
image to a square with an ordinary bicubic resize, which is reproducible
exactly, and its 576 patches cost far less per query than OWLv2's 3600.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import numpy as np
from PIL import Image

from app.core.errors import BadRequestError, ServiceUnavailableError
from app.detection.base import DetectorProvider
from app.schemas.image import CropRectangle, DetectedRegion
from app.utils.onnx_utils import (
    integer_dtype_of,
    load_clip_tokenizer,
    load_session,
    trailing_dimension_of,
)

#: Used only when an export leaves the image dimensions dynamic.
DEFAULT_INPUT_SIZE = 768

#: Prompt token budget. OWL-ViT pads every prompt to this length.
DEFAULT_CONTEXT_LENGTH = 16

#: OWL-ViT inherits CLIP's training normalisation.
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

#: Overlap above which two boxes are treated as the same object.
NMS_IOU_THRESHOLD = 0.3


class OWLViTONNXProvider(DetectorProvider):
    """Open-vocabulary detector served by ONNX Runtime.

    The session loads lazily on first use and is reused for the process
    lifetime, so keep one provider instance per model.
    """

    provider_name = "owlvit"

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        if model_path is None or tokenizer_path is None:
            raise ValueError(
                "The owlvit detector needs DETECTOR_MODEL_PATH and "
                "DETECTOR_TOKENIZER_PATH; build them with "
                "scripts/export_detector_onnx.py"
            )

        self._model_path = Path(model_path)
        self._tokenizer_path = Path(tokenizer_path)

        self._input_size = DEFAULT_INPUT_SIZE
        self._context_length = DEFAULT_CONTEXT_LENGTH
        self._session: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._end_of_text_id: Optional[int] = None
        self._token_dtype = np.int64
        self._load_lock = Lock()

    def detect(
        self,
        image: Image.Image,
        *,
        prompts: Sequence[str],
        max_regions: int,
        min_score: float,
    ) -> list[DetectedRegion]:
        if not prompts:
            raise BadRequestError(
                message=(
                    "Open-vocabulary detection needs at least one prompt saying "
                    "what to look for"
                ),
                code="DETECTOR_PROMPTS_REQUIRED",
            )
        if max_regions < 1:
            return []

        self._ensure_loaded()

        try:
            pixels = self._to_input_array(image)
            tokens, attention_mask = self._tokenize(list(prompts))
            logits, boxes = self._run(pixels, tokens, attention_mask)
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not run the object detector",
                code="DETECTOR_INFERENCE_FAILED",
                details={"model": str(self._model_path), "error": str(exc)},
            ) from exc

        return self._regions_from_outputs(
            logits=logits,
            boxes=boxes,
            prompts=list(prompts),
            max_regions=max_regions,
            min_score=min_score,
        )

    def warmup(self) -> None:
        self._ensure_loaded()

    # --- Preprocessing -------------------------------------------------

    def _to_input_array(self, image: Image.Image) -> np.ndarray:
        """Square the image the way OWL-ViT does, then normalize.

        OWL-ViT stretches an image to its square input rather than padding or
        cropping it, which is the one detail that makes the boxes easy to use:
        scaling each axis independently leaves normalized coordinates unchanged,
        so what the model predicts is already relative to the caller's image.
        """
        rgb_image = image.convert("RGB") if image.mode != "RGB" else image.copy()
        try:
            if rgb_image.width == 0 or rgb_image.height == 0:
                raise ValueError("image has no pixels")

            resized = rgb_image.resize(
                (self._input_size, self._input_size), Image.BICUBIC
            )
            try:
                array = np.asarray(resized, dtype=np.float32) / 255.0
            finally:
                resized.close()
        finally:
            rgb_image.close()

        array = (array - np.asarray(IMAGE_MEAN, dtype=np.float32)) / np.asarray(
            IMAGE_STD, dtype=np.float32
        )
        return np.transpose(array, (2, 0, 1))[np.newaxis, ...]

    def _tokenize(self, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Encode prompts the way OWL-ViT's reference processor does.

        Padding is the end-of-text token rather than zero, and the attention
        mask is what actually tells the model to ignore it.
        """
        encodings = self._tokenizer.encode_batch(prompts)
        context_length = self._context_length

        rows, masks = [], []
        for encoding in encodings:
            ids = list(encoding.ids[:context_length])
            # A truncated prompt must still end in end-of-text: the text tower
            # reads its output at that position.
            if len(ids) == context_length:
                ids[-1] = self._end_of_text_id

            mask = [1] * len(ids)
            padding = context_length - len(ids)
            ids.extend([self._end_of_text_id] * padding)
            mask.extend([0] * padding)

            rows.append(ids)
            masks.append(mask)

        return (
            np.asarray(rows, dtype=self._token_dtype),
            np.asarray(masks, dtype=self._token_dtype),
        )

    # --- Inference -----------------------------------------------------

    def _run(
        self, pixels: np.ndarray, tokens: np.ndarray, attention_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Feed only the inputs this particular export actually declares."""
        available = {
            "pixel_values": pixels,
            "input_ids": tokens,
            "attention_mask": attention_mask,
        }
        feed = {
            model_input.name: available[model_input.name]
            for model_input in self._session.get_inputs()
            if model_input.name in available
        }

        output_names = [output.name for output in self._session.get_outputs()]
        outputs = self._session.run(None, feed)
        named = dict(zip(output_names, outputs))

        logits = named.get("logits", outputs[0])
        boxes = named.get("pred_boxes", outputs[-1])
        return np.asarray(logits, np.float32), np.asarray(boxes, np.float32)

    def _regions_from_outputs(
        self,
        *,
        logits: np.ndarray,
        boxes: np.ndarray,
        prompts: list[str],
        max_regions: int,
        min_score: float,
    ) -> list[DetectedRegion]:
        # One image per call, so drop the batch axis.
        scores_per_prompt = _sigmoid(logits[0])
        best_prompt = scores_per_prompt.argmax(axis=-1)
        scores = scores_per_prompt.max(axis=-1)

        keep = np.flatnonzero(scores >= min_score)
        if keep.size == 0:
            return []

        corners = np.clip(_center_to_corners(boxes[0][keep]), 0.0, 1.0)
        scores = scores[keep]
        labels = best_prompt[keep]

        # Strongest first, so non-maximum suppression keeps the best of a cluster.
        order = np.argsort(-scores)
        kept = _suppress_overlaps(corners[order], NMS_IOU_THRESHOLD, max_regions)

        regions = []
        for index in kept:
            source_index = order[index]
            x1, y1, x2, y2 = corners[source_index]
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            regions.append(
                DetectedRegion(
                    box=CropRectangle(
                        x=float(x1),
                        y=float(y1),
                        width=float(x2 - x1),
                        height=float(y2 - y1),
                    ),
                    score=float(scores[source_index]),
                    label=prompts[int(labels[source_index])],
                )
            )

        return regions

    # --- Loading -------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return

        with self._load_lock:
            if self._session is None:
                tokenizer, end_of_text_id = load_clip_tokenizer(
                    self._tokenizer_path,
                    failed_code="DETECTOR_TOKENIZER_LOAD_FAILED",
                )
                session = load_session(
                    self._model_path,
                    missing_code="DETECTOR_MODEL_LOAD_FAILED",
                    failed_code="DETECTOR_MODEL_LOAD_FAILED",
                )
                self._tokenizer = tokenizer
                self._end_of_text_id = end_of_text_id
                self._input_size = self._dimension_of(
                    session, "pixel_values", -2, DEFAULT_INPUT_SIZE
                )
                self._context_length = self._dimension_of(
                    session, "input_ids", -1, DEFAULT_CONTEXT_LENGTH
                )
                self._token_dtype = self._token_dtype_of(session)
                # Assigned last: a non-None session means fully loaded.
                self._session = session

    @staticmethod
    def _dimension_of(session: Any, name: str, axis: int, default: int) -> int:
        index = _input_index(session, name)
        if index is None:
            return default
        return trailing_dimension_of(session, index=index, axis=axis, default=default)

    @staticmethod
    def _token_dtype_of(session: Any) -> Any:
        index = _input_index(session, "input_ids")
        # `index or 0` would read the same but is the classic falsy-zero trap;
        # input_ids being the first input is the common case, not a missing one.
        return integer_dtype_of(session, index=0 if index is None else index)


def _input_index(session: Any, name: str) -> Optional[int]:
    for index, model_input in enumerate(session.get_inputs()):
        if model_input.name == name:
            return index
    return None


def _sigmoid(values: np.ndarray) -> np.ndarray:
    # Split by sign so neither branch overflows exp on an extreme logit.
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _center_to_corners(boxes: np.ndarray) -> np.ndarray:
    """OWL-ViT predicts centre-x, centre-y, width, height."""
    centre_x, centre_y, width, height = boxes.T
    return np.stack(
        [
            centre_x - width / 2,
            centre_y - height / 2,
            centre_x + width / 2,
            centre_y + height / 2,
        ],
        axis=-1,
    )


def _suppress_overlaps(corners: np.ndarray, iou_threshold: float, limit: int) -> list:
    """Greedy non-maximum suppression over boxes already sorted by score."""
    areas = (corners[:, 2] - corners[:, 0]) * (corners[:, 3] - corners[:, 1])
    kept: list = []

    for index in range(len(corners)):
        if len(kept) >= limit:
            break
        overlaps_a_kept_box = any(
            _intersection_over_union(
                corners[index], corners[other], areas[index], areas[other]
            )
            > iou_threshold
            for other in kept
        )
        if not overlaps_a_kept_box:
            kept.append(index)

    return kept


def _intersection_over_union(
    first: np.ndarray, second: np.ndarray, first_area: float, second_area: float
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])

    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    union = first_area + second_area - overlap
    return float(overlap / union) if union > 0 else 0.0
