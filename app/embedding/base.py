from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from app.core.errors import BadRequestError

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class EmbeddingModelMetadata:
    provider: str
    model_name: str
    model_pretrained: str
    vector_size: int
    supports_text: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EmbeddingProvider(ABC):
    """Contract every embedding backend implements.

    A collection is pinned to one provider and model, because vectors from
    different models cannot be compared.
    """

    provider_name: ClassVar[str]

    #: True when the provider maps text into the same vector space as its
    #: images, which is what makes text and hybrid search possible. Normally a
    #: fact about the class, but a provider whose text encoder is a separate,
    #: optional artifact narrows it per instance.
    supports_text: bool = False

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
            supports_text=self.supports_text,
        )

    @abstractmethod
    def embed_image(self, image: Image.Image) -> list[float]:
        """Return an embedding vector for a single image."""

    def embed_images(self, images: Sequence[Image.Image]) -> list[list[float]]:
        """Return one embedding vector per image, in order.

        The default walks the images one by one. Providers backed by a tensor
        library should override this with a real batched forward pass.
        """
        return [self.embed_image(image) for image in images]

    def embed_text(self, text: str) -> list[float]:
        """Return an embedding vector for a text query.

        Only meaningful when the provider's text and image encoders share one
        vector space, so the base implementation refuses rather than returning
        a vector that cannot be compared with the stored image vectors.
        """
        raise BadRequestError(
            message=f"The {self.provider_name} provider does not support text queries",
            code="TEXT_SEARCH_NOT_SUPPORTED",
            details={"provider": self.provider_name},
        )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per text, in order."""
        return [self.embed_text(text) for text in texts]

    def warmup(self) -> None:  # noqa: B027 - optional hook, not every provider needs it
        """Load model weights eagerly. Safe to call more than once."""
