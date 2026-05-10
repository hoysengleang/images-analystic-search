from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class EmbeddingModelMetadata:
    provider: str
    model_name: str
    model_pretrained: str
    vector_size: int

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


class EmbeddingProvider(ABC):
    provider_name: ClassVar[str]

    def __init__(
        self,
        *,
        model_name: str,
        model_pretrained: str,
        vector_size: int,
    ) -> None:
        self.model_name = model_name
        self.model_pretrained = model_pretrained
        self._vector_size = vector_size

    @property
    def vector_size(self) -> int:
        return self._vector_size

    @property
    def metadata(self) -> EmbeddingModelMetadata:
        return EmbeddingModelMetadata(
            provider=self.provider_name,
            model_name=self.model_name,
            model_pretrained=self.model_pretrained,
            vector_size=self.vector_size,
        )

    @abstractmethod
    def embed_image(self, image: "Image.Image") -> list[float]:
        """Return an embedding vector for an image."""
