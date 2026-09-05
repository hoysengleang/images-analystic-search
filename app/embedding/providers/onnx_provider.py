"""ONNX Runtime embedding provider.

Runs an exported CLIP encoder through onnxruntime instead of PyTorch. The
vectors are the same; the dependency is tens of megabytes rather than the best
part of a gigabyte, which is what makes a small self-hosted deployment
practical. Produce the model files with ``scripts/export_onnx.py``.

Preprocessing is reimplemented here on Pillow and NumPy, because the transform
OpenCLIP hands back is a torchvision object that does not exist without
PyTorch. It performs the same steps in the same order, and the export script
asserts that the two agree numerically before it writes anything out.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import numpy as np
from PIL import Image

from app.core.errors import BadRequestError, ServiceUnavailableError
from app.embedding.base import EmbeddingProvider
from app.utils.image_utils import (
    center_crop_fraction,
    image_to_rgb,
    letterbox_to_square,
    resize_shortest_side_and_center_crop,
)
from app.utils.onnx_utils import (
    integer_dtype_of,
    load_clip_tokenizer,
    load_session,
    trailing_dimension_of,
)

#: How a non-square image is fitted to the model's square input.
#: "crop" is the stock CLIP transform; "pad" keeps the whole frame.
FRAMING_MODES = ("crop", "pad")

#: Used only when an export leaves height and width dynamic.
DEFAULT_INPUT_SIZE = 224

#: CLIP's training normalisation, shared by every open_clip CLIP checkpoint.
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

#: Token positions in CLIP's text encoder. open_clip pads with zeros rather
#: than with the end-of-text id, so a matching export expects zeros too.
CONTEXT_LENGTH = 77
PAD_TOKEN_ID = 0


class ONNXProvider(EmbeddingProvider):
    """CLIP encoder served by ONNX Runtime, producing L2-normalized vectors.

    The image encoder is required. The text encoder is optional: without it the
    provider reports ``supports_text = False`` and text and hybrid search are
    refused, which keeps a deployment that only searches by image from having
    to ship a second model. Sessions load lazily on first use and are reused
    for the process lifetime, so keep one provider instance per model.
    """

    provider_name = "onnx"

    def __init__(
        self,
        *,
        model_name: str,
        model_pretrained: str,
        vector_size: int,
        image_model_path: Optional[str] = None,
        text_model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        framing: str = "crop",
        views: int = 1,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_pretrained=model_pretrained,
            vector_size=vector_size,
        )
        # Declared Optional so the manager, which drops unset options, reaches
        # this message instead of a bare TypeError about a missing argument.
        if image_model_path is None:
            raise ValueError(
                "The onnx provider needs ONNX_IMAGE_MODEL_PATH to point at an "
                "exported image encoder; build one with scripts/export_onnx.py"
            )
        if text_model_path is not None and tokenizer_path is None:
            raise ValueError(
                "ONNX_TEXT_MODEL_PATH also needs ONNX_TOKENIZER_PATH; the text "
                "encoder cannot run without the tokenizer it was exported with"
            )
        if framing not in FRAMING_MODES:
            raise ValueError(f"framing must be one of {FRAMING_MODES}, got {framing!r}")
        if not 1 <= views <= 3:
            raise ValueError(f"views must be between 1 and 3, got {views}")

        self.framing = framing
        self.views = views
        # Capability follows configuration here, unlike OpenCLIP where the one
        # checkpoint always carries both encoders.
        self.supports_text = text_model_path is not None

        self._image_model_path = Path(image_model_path)
        self._text_model_path = Path(text_model_path) if text_model_path else None
        self._tokenizer_path = Path(tokenizer_path) if tokenizer_path else None

        self._input_size = DEFAULT_INPUT_SIZE
        self._image_session: Optional[Any] = None
        self._text_session: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._end_of_text_id: Optional[int] = None
        self._token_dtype = np.int64
        self._load_lock = Lock()

    def embed_image(self, image: Image.Image) -> list[float]:
        return self.embed_images([image])[0]

    def embed_images(self, images: Sequence[Image.Image]) -> list[list[float]]:
        """Encode a batch of images in a single session run."""
        if not images:
            return []

        self._ensure_image_session()

        try:
            view_count = self.views
            prepared_views: list[Image.Image] = []
            try:
                for image in images:
                    prepared_views.extend(self._views_of(image))
                batch = np.stack([self._to_input_array(v) for v in prepared_views])
            finally:
                for view in prepared_views:
                    view.close()

            features = self._normalize(self._run(self._image_session, batch))
            if view_count > 1:
                # Average the views back into one vector per image, then
                # renormalize so the result stays comparable to the rest.
                features = features.reshape(len(images), view_count, -1).mean(axis=1)
                features = self._normalize(features)

            vectors = features.tolist()
        except Exception as exc:
            raise self._encoding_error("ONNX_IMAGE_EMBEDDING_FAILED", exc) from exc

        return self._validated_vectors(vectors)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a batch of text queries in a single session run."""
        if not texts:
            return []
        if self._text_model_path is None:
            raise BadRequestError(
                message=(
                    "This ONNX deployment has no text encoder; set "
                    "ONNX_TEXT_MODEL_PATH to search by text"
                ),
                code="TEXT_SEARCH_NOT_SUPPORTED",
                details={"provider": self.provider_name},
            )

        self._ensure_text_session()

        try:
            tokens = self._tokenize(list(texts))
            features = self._normalize(self._run(self._text_session, tokens))
            vectors = features.tolist()
        except Exception as exc:
            raise self._encoding_error("ONNX_TEXT_EMBEDDING_FAILED", exc) from exc

        return self._validated_vectors(vectors)

    def warmup(self) -> None:
        self._ensure_image_session()
        if self._text_model_path is not None:
            self._ensure_text_session()

    # --- Preprocessing -------------------------------------------------

    def _views_of(self, image: Image.Image) -> list[Image.Image]:
        """Frame one image, then derive the extra views used for averaging.

        Deliberately identical to the OpenCLIP provider, including the fact
        that "crop" hands the full-resolution image on untouched: the stock
        transform frames at the very end, so the extra views below are taken
        from the whole photo rather than from an already-cropped square. Doing
        it in the other order silently narrows the field of view.
        """
        if self.framing == "pad":
            framed = letterbox_to_square(image, self._input_size)
        else:
            framed = image_to_rgb(image)

        views = [framed]
        if self.views >= 2:
            views.append(center_crop_fraction(framed, 0.85))
        if self.views >= 3:
            views.append(framed.transpose(Image.FLIP_LEFT_RIGHT))
        return views

    def _to_input_array(self, image: Image.Image) -> np.ndarray:
        """Turn one view into the normalized CHW tensor CLIP expects.

        The NumPy equivalent of the torchvision transform OpenCLIP hands back,
        applied to every view for the same reason it is there: an already
        letterboxed frame passes through untouched, while a raw or cropped
        view is framed here.
        """
        framed = resize_shortest_side_and_center_crop(image, self._input_size)
        try:
            array = np.asarray(framed, dtype=np.float32) / 255.0
        finally:
            framed.close()

        array = (array - np.asarray(IMAGE_MEAN, dtype=np.float32)) / np.asarray(
            IMAGE_STD, dtype=np.float32
        )
        return np.transpose(array, (2, 0, 1))

    def _tokenize(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)

        rows = []
        for encoding in encodings:
            ids = list(encoding.ids[:CONTEXT_LENGTH])
            # A truncated sequence still has to end in end-of-text: the encoder
            # reads its output at that position, so dropping it would quietly
            # return the embedding of whichever token happened to land last.
            if len(ids) == CONTEXT_LENGTH:
                ids[-1] = self._end_of_text_id
            ids.extend([PAD_TOKEN_ID] * (CONTEXT_LENGTH - len(ids)))
            rows.append(ids)

        return np.asarray(rows, dtype=self._token_dtype)

    # --- Inference -----------------------------------------------------

    @staticmethod
    def _run(session: Any, batch: np.ndarray) -> np.ndarray:
        outputs = session.run(None, {session.get_inputs()[0].name: batch})
        return np.asarray(outputs[0], dtype=np.float32)

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        # A zero-length vector cannot be normalized; leaving it as zeros beats
        # dividing by zero and writing NaNs into the index.
        return features / np.maximum(norms, np.finfo(np.float32).tiny)

    def _validated_vectors(self, vectors: Sequence[Sequence[float]]) -> list[list[float]]:
        for vector in vectors:
            if len(vector) != self.vector_size:
                raise ServiceUnavailableError(
                    message="ONNX vector size does not match provider configuration",
                    code="ONNX_VECTOR_SIZE_MISMATCH",
                    details={
                        "expected_vector_size": self.vector_size,
                        "actual_vector_size": len(vector),
                    },
                )

        return [[float(value) for value in vector] for vector in vectors]

    def _encoding_error(self, code: str, exc: Exception) -> ServiceUnavailableError:
        return ServiceUnavailableError(
            message="Could not encode input with ONNX Runtime",
            code=code,
            details={
                "model": self.model_name,
                "pretrained": self.model_pretrained,
                "error": str(exc),
            },
        )

    # --- Loading -------------------------------------------------------

    def _ensure_image_session(self) -> None:
        if self._image_session is not None:
            return

        with self._load_lock:
            if self._image_session is None:
                session = self._load_session(self._image_model_path)
                self._input_size = self._input_size_of(session)
                # Assigned last: a non-None session means fully loaded.
                self._image_session = session

    def _ensure_text_session(self) -> None:
        if self._text_session is not None:
            return

        with self._load_lock:
            if self._text_session is None:
                tokenizer, end_of_text_id = self._load_tokenizer()
                session = self._load_session(self._text_model_path)
                self._tokenizer = tokenizer
                self._end_of_text_id = end_of_text_id
                self._token_dtype = self._token_dtype_of(session)
                self._text_session = session

    def _load_session(self, model_path: Path) -> Any:
        return load_session(
            model_path,
            missing_code="ONNX_MODEL_LOAD_FAILED",
            failed_code="ONNX_MODEL_LOAD_FAILED",
        )

    def _load_tokenizer(self) -> tuple[Any, int]:
        return load_clip_tokenizer(
            self._tokenizer_path,
            failed_code="ONNX_TOKENIZER_LOAD_FAILED",
        )

    @staticmethod
    def _token_dtype_of(session: Any) -> Any:
        return integer_dtype_of(session)

    @staticmethod
    def _input_size_of(session: Any) -> int:
        """Read the square input edge the model was exported with."""
        return trailing_dimension_of(
            session, index=0, axis=-2, default=DEFAULT_INPUT_SIZE
        )
