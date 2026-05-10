from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any, Optional

from app.core.errors import ServiceUnavailableError
from app.embedding.base import EmbeddingProvider
from app.utils.image_utils import image_to_rgb

if TYPE_CHECKING:
    from PIL import Image


class OpenCLIPProvider(EmbeddingProvider):
    provider_name = "openclip"

    def __init__(
        self,
        *,
        model_name: str,
        model_pretrained: str,
        vector_size: int,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_pretrained=model_pretrained,
            vector_size=vector_size,
        )
        self._model: Optional[Any] = None
        self._preprocess: Optional[Any] = None
        self._device: Optional[Any] = None
        self._load_lock = Lock()

    def embed_image(self, image: "Image.Image") -> list[float]:
        model, preprocess, device = self._get_model()

        try:
            import torch

            rgb_image = image_to_rgb(image)
            image_tensor = preprocess(rgb_image).unsqueeze(0).to(device)

            with torch.inference_mode():
                image_features = model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(
                    dim=-1,
                    keepdim=True,
                )

            vector = image_features.squeeze(0).detach().cpu().float().tolist()
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not encode image with OpenCLIP",
                code="OPENCLIP_IMAGE_EMBEDDING_FAILED",
                details={"model": self.model_name, "pretrained": self.model_pretrained},
            ) from exc

        if len(vector) != self.vector_size:
            raise ServiceUnavailableError(
                message="OpenCLIP vector size does not match provider configuration",
                code="OPENCLIP_VECTOR_SIZE_MISMATCH",
                details={
                    "expected_vector_size": self.vector_size,
                    "actual_vector_size": len(vector),
                },
            )

        return [float(value) for value in vector]

    def _get_model(self) -> tuple[Any, Any, Any]:
        if self._model is not None and self._preprocess is not None:
            return self._model, self._preprocess, self._device

        with self._load_lock:
            if self._model is None or self._preprocess is None:
                self._load_model()

        return self._model, self._preprocess, self._device

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
        except Exception as exc:
            raise ServiceUnavailableError(
                message="Could not load OpenCLIP model",
                code="OPENCLIP_MODEL_LOAD_FAILED",
                details={"model": self.model_name, "pretrained": self.model_pretrained},
            ) from exc

        self._model = model
        self._preprocess = preprocess
        self._device = device
