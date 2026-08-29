from __future__ import annotations

from collections.abc import Sequence
from threading import Lock
from typing import TYPE_CHECKING, Any, Optional

from app.core.errors import ServiceUnavailableError
from app.embedding.base import EmbeddingProvider
from app.utils.image_utils import (
    center_crop_fraction,
    image_to_rgb,
    letterbox_to_square,
)

#: How a non-square image is fitted to the model's square input.
#: "crop" is the stock CLIP transform; "pad" keeps the whole frame.
FRAMING_MODES = ("crop", "pad")

DEFAULT_INPUT_SIZE = 224

if TYPE_CHECKING:
    from PIL import Image


class OpenCLIPProvider(EmbeddingProvider):
    """OpenCLIP encoder producing L2-normalized vectors.

    Images and text are encoded into the same vector space, so a text query can
    be compared directly against stored image vectors. Weights load lazily on
    first use and are reused for the process lifetime, so keep one provider
    instance per model.
    """

    provider_name = "openclip"
    supports_text = True

    def __init__(
        self,
        *,
        model_name: str,
        model_pretrained: str,
        vector_size: int,
        framing: str = "crop",
        views: int = 1,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_pretrained=model_pretrained,
            vector_size=vector_size,
        )
        if framing not in FRAMING_MODES:
            raise ValueError(f"framing must be one of {FRAMING_MODES}, got {framing!r}")
        if not 1 <= views <= 3:
            raise ValueError(f"views must be between 1 and 3, got {views}")

        self.framing = framing
        self.views = views
        self._input_size = DEFAULT_INPUT_SIZE
        self._model: Optional[Any] = None
        self._preprocess: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._device: Optional[Any] = None
        self._load_lock = Lock()

    def embed_image(self, image: Image.Image) -> list:
        return self.embed_images([image])[0]

    def embed_images(self, images: Sequence) -> list:
        """Encode a batch of images in a single forward pass."""
        if not images:
            return []

        self._ensure_loaded()

        try:
            import torch

            # Every view of every image goes through one forward pass, so extra
            # views cost GPU/CPU work but no extra round trips.
            view_count = self.views
            prepared_views = []
            try:
                for image in images:
                    prepared_views.extend(self._views_of(image))
                batch = torch.stack(
                    [self._preprocess(view) for view in prepared_views]
                ).to(self._device)
            finally:
                for view in prepared_views:
                    view.close()

            with torch.inference_mode():
                features = self._model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
                if view_count > 1:
                    # Average the views back into one vector per image, then
                    # renormalize so the result stays comparable to the rest.
                    features = features.reshape(len(images), view_count, -1).mean(dim=1)
                    features = features / features.norm(dim=-1, keepdim=True)

            vectors = features.detach().cpu().float().tolist()
        except Exception as exc:
            raise self._encoding_error("OPENCLIP_IMAGE_EMBEDDING_FAILED", exc) from exc

        return self._validated_vectors(vectors)

    def _views_of(self, image) -> list:
        """Frame one image, then derive the extra views used for averaging."""
        if self.framing == "pad":
            framed = letterbox_to_square(image, self._input_size)
        else:
            framed = image_to_rgb(image)

        views = [framed]
        if self.views >= 2:
            views.append(center_crop_fraction(framed, 0.85))
        if self.views >= 3:
            from PIL import Image as PILImage

            views.append(framed.transpose(PILImage.FLIP_LEFT_RIGHT))
        return views

    def embed_text(self, text: str) -> list:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence) -> list:
        """Encode a batch of text queries in a single forward pass."""
        if not texts:
            return []

        self._ensure_loaded()

        try:
            import torch

            tokens = self._tokenizer(list(texts)).to(self._device)

            with torch.inference_mode():
                features = self._model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)

            vectors = features.detach().cpu().float().tolist()
        except Exception as exc:
            raise self._encoding_error("OPENCLIP_TEXT_EMBEDDING_FAILED", exc) from exc

        return self._validated_vectors(vectors)

    def warmup(self) -> None:
        self._ensure_loaded()

    def _validated_vectors(self, vectors: Sequence) -> list:
        for vector in vectors:
            if len(vector) != self.vector_size:
                raise ServiceUnavailableError(
                    message="OpenCLIP vector size does not match provider configuration",
                    code="OPENCLIP_VECTOR_SIZE_MISMATCH",
                    details={
                        "expected_vector_size": self.vector_size,
                        "actual_vector_size": len(vector),
                    },
                )

        return [[float(value) for value in vector] for vector in vectors]

    def _encoding_error(self, code: str, exc: Exception) -> ServiceUnavailableError:
        return ServiceUnavailableError(
            message="Could not encode input with OpenCLIP",
            code=code,
            details={
                "model": self.model_name,
                "pretrained": self.model_pretrained,
                "error": str(exc),
            },
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is None:
                self._load_model()

    def _load_model(self) -> None:
        try:
            import open_clip
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.model_pretrained,
                device=device,
            )
            model.eval()
            tokenizer = open_clip.get_tokenizer(self.model_name)
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not load OpenCLIP model",
                code="OPENCLIP_MODEL_LOAD_FAILED",
                details={"model": self.model_name, "pretrained": self.model_pretrained},
            ) from exc

        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._device = device
        # Assigned last: _ensure_loaded treats a non-None model as fully loaded.
        self._model = model
