from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from app.schemas.image import DetectedRegion

if TYPE_CHECKING:
    from PIL import Image


class DetectorProvider(ABC):
    """Contract every object detector implements.

    A detector narrows a *query* image before it is embedded: a shopper's photo
    of a bag on a cluttered table becomes a photo of a bag. It never touches
    stored vectors, so a deployment can turn detection on or off without
    reindexing anything.
    """

    provider_name: ClassVar[str]

    @abstractmethod
    def detect(
        self,
        image: Image.Image,
        *,
        prompts: Sequence[str],
        max_regions: int,
        min_score: float,
    ) -> list[DetectedRegion]:
        """Return the regions worth searching, strongest first.

        An empty list is a normal answer, not an error: the caller falls back to
        the whole image. Boxes are normalized to the image passed in, so the
        caller never has to know the detector's own input size.
        """

    def warmup(self) -> None: 
        """Load model weights eagerly. Safe to call more than once."""
