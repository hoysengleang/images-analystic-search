from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.constants import (
    EXTENSION_ALIASES,
    PIL_FORMAT_TO_EXTENSION,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from app.core.errors import (
    BadRequestError,
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedImageTypeError,
)

#: A crop shorter than this on either edge carries too few pixels to embed
#: usefully, and in practice means the caller sent pixels where the API asks for
#: fractions of the image.
MIN_CROP_PIXELS = 16


@dataclass(frozen=True)
class ValidatedImage:
    image: Image.Image
    format: str
    extension: str
    size_bytes: int
    width: int
    height: int


def normalize_image_extension(extension_or_path: str) -> str:
    """Return the canonical lowercase extension for a path or extension."""
    suffix = Path(extension_or_path).suffix
    extension = suffix if suffix else extension_or_path
    normalized = extension.strip().lower().lstrip(".")
    return EXTENSION_ALIASES.get(normalized, normalized)


def validate_image_extension(
    extension_or_path: str,
    *,
    allowed_extensions: Optional[Iterable[str]] = None,
) -> str:
    """Return the canonical extension, or raise if it is not accepted."""
    extension = normalize_image_extension(extension_or_path)
    allowed = frozenset(allowed_extensions or SUPPORTED_IMAGE_EXTENSIONS)

    if extension not in allowed:
        raise UnsupportedImageTypeError(
            message="Unsupported image file extension",
            details={
                "extension": extension_or_path,
                "allowed_extensions": sorted(allowed),
            },
        )

    return extension


def validate_and_load_image(
    image_bytes: bytes,
    *,
    max_size_mb: int,
    extension: Optional[str] = None,
    allowed_extensions: Optional[Iterable[str]] = None,
    max_pixels: Optional[int] = None,
) -> ValidatedImage:
    """Decode ``image_bytes`` into an RGB image after every safety check.

    The declared ``extension`` (from a filename or URL path) is only trusted
    far enough to reject it early; the real format always comes from Pillow.
    """
    allowed = frozenset(allowed_extensions or SUPPORTED_IMAGE_EXTENSIONS)
    _validate_image_size(image_bytes, max_size_mb=max_size_mb)

    declared_extension = (
        validate_image_extension(extension, allowed_extensions=allowed)
        if extension is not None
        else None
    )

    image_format = _read_image_format(image_bytes, max_pixels=max_pixels)
    actual_extension = PIL_FORMAT_TO_EXTENSION.get(image_format)

    if actual_extension is None or actual_extension not in allowed:
        raise UnsupportedImageTypeError(
            message="Unsupported image format",
            details={
                "format": image_format or None,
                "allowed_extensions": sorted(allowed),
            },
        )

    if declared_extension is not None and declared_extension != actual_extension:
        raise UnsupportedImageTypeError(
            message="Image file extension does not match image format",
            details={"extension": declared_extension, "format": actual_extension},
        )

    loaded_image = _load_rgb_image(image_bytes)

    return ValidatedImage(
        image=loaded_image,
        format=image_format,
        extension=actual_extension,
        size_bytes=len(image_bytes),
        width=loaded_image.width,
        height=loaded_image.height,
    )


def letterbox_to_square(
    image: Image.Image,
    size: int,
    *,
    fill: tuple = (124, 116, 104),
) -> Image.Image:
    """Fit the whole image into a square without cropping anything away.

    The usual CLIP transform resizes the short side then centre-crops, which
    silently discards the top and bottom of a portrait photo — around 40% of a
    3:5 phone picture. Padding keeps the entire subject in frame at the cost of
    two neutral bars. ``fill`` defaults to the CLIP training mean, which
    normalizes to roughly zero and so disturbs the model least.
    """
    rgb_image = image_to_rgb(image)
    scaled: Optional[Image.Image] = None
    try:
        width, height = rgb_image.size
        if width == 0 or height == 0:
            raise InvalidImageError(message="Image has no pixels")

        scale = size / max(width, height)
        scaled_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        scaled = rgb_image.resize(scaled_size, Image.BICUBIC)

        canvas = Image.new("RGB", (size, size), fill)
        canvas.paste(
            scaled,
            ((size - scaled_size[0]) // 2, (size - scaled_size[1]) // 2),
        )
        return canvas
    finally:
        if scaled is not None:
            scaled.close()
        rgb_image.close()


def resize_shortest_side_and_center_crop(image: Image.Image, size: int) -> Image.Image:
    """Fit the image to a square the way the stock CLIP transform does.

    The shortest side is scaled to ``size`` and the middle square is kept, so
    whatever falls outside that square is discarded. See
    :func:`letterbox_to_square` for the alternative that keeps the whole frame.

    Reimplemented on Pillow so a provider that does not depend on torchvision
    can still reproduce its output. The mixed arithmetic below looks arbitrary
    and is not: torchvision truncates the scaled long edge but rounds the crop
    offset, and copying that exactly is what keeps the two byte-identical.
    Rounding both the obvious way shifts the crop by a pixel on most landscape
    photos, which is invisible until search quality drifts from the reference.
    """
    rgb_image = image_to_rgb(image)
    scaled: Optional[Image.Image] = None
    try:
        width, height = rgb_image.size
        if width == 0 or height == 0:
            raise InvalidImageError(message="Image has no pixels")

        # The shortest side lands on `size` exactly, so the long side is always
        # at least `size` and the crop below never runs off the edge.
        short, long = (width, height) if width <= height else (height, width)
        scaled_long = int(size * long / short)
        scaled_size = (size, scaled_long) if width <= height else (scaled_long, size)
        scaled = rgb_image.resize(scaled_size, Image.BICUBIC)

        left = int(round((scaled_size[0] - size) / 2.0))
        top = int(round((scaled_size[1] - size) / 2.0))
        cropped = scaled.crop((left, top, left + size, top + size))
        # crop() is lazy and keeps a reference to its source, so the pixels have
        # to be materialized before that source is closed below.
        cropped.load()
        return cropped
    finally:
        if scaled is not None:
            scaled.close()
        rgb_image.close()


def crop_to_normalized_box(
    image: Image.Image,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    min_pixels: int = MIN_CROP_PIXELS,
) -> Image.Image:
    """Cut out the region a caller asked for, given in fractions of the image.

    Normalized coordinates let one box serve a thumbnail and the full-resolution
    original alike, which is why the API takes them that way. The box is clamped
    to the image rather than refused when rounding pushes an edge a pixel past
    the boundary; only a box with almost no pixels in it is an error, because
    that is what a caller who sent pixel coordinates by mistake produces.
    """
    rgb_image = image_to_rgb(image)
    try:
        image_width, image_height = rgb_image.size
        if image_width == 0 or image_height == 0:
            raise InvalidImageError(message="Image has no pixels")

        left = _clamp(round(x * image_width), 0, image_width - 1)
        top = _clamp(round(y * image_height), 0, image_height - 1)
        right = _clamp(round((x + width) * image_width), left + 1, image_width)
        bottom = _clamp(round((y + height) * image_height), top + 1, image_height)

        if right - left < min_pixels or bottom - top < min_pixels:
            raise BadRequestError(
                message="Crop rectangle covers too little of the image to search",
                code="CROP_TOO_SMALL",
                details={
                    "crop_width_pixels": right - left,
                    "crop_height_pixels": bottom - top,
                    "min_pixels": min_pixels,
                    "image_width": image_width,
                    "image_height": image_height,
                },
            )

        cropped = rgb_image.crop((left, top, right, bottom))
        # crop() is lazy and keeps a reference to its source, so materialize the
        # pixels before the RGB copy below is closed.
        cropped.load()
        return cropped
    finally:
        rgb_image.close()


def _clamp(value: int, lowest: int, highest: int) -> int:
    return max(lowest, min(value, highest))


def center_crop_fraction(image: Image.Image, fraction: float) -> Image.Image:
    """Return the middle ``fraction`` of the image, used for extra views."""
    width, height = image.size
    crop_w, crop_h = int(width * fraction), int(height * fraction)
    left, top = (width - crop_w) // 2, (height - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h))


def image_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    return image.convert("RGB")


def _validate_image_size(image_bytes: bytes, *, max_size_mb: int) -> None:
    if max_size_mb < 1:
        raise ValueError("max_size_mb must be at least 1")

    max_size_bytes = max_size_mb * 1024 * 1024
    if len(image_bytes) > max_size_bytes:
        raise ImageTooLargeError(
            details={
                "size_bytes": len(image_bytes),
                "max_size_bytes": max_size_bytes,
            },
        )


def _read_image_format(image_bytes: bytes, *, max_pixels: Optional[int]) -> str:
    """Verify the payload really is an image and return its Pillow format.

    The pixel count is checked from the header before any decoding, so a small
    file that expands into an enormous bitmap is refused rather than decoded.
    """
    try:
        with Image.open(BytesIO(image_bytes)) as opened_image:
            image_format = (opened_image.format or "").upper()
            width, height = opened_image.size
            opened_image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(message="Invalid or broken image file") from exc

    if max_pixels is not None and width * height > max_pixels:
        raise ImageTooLargeError(
            message="Image has too many pixels",
            details={
                "width": width,
                "height": height,
                "pixels": width * height,
                "max_pixels": max_pixels,
            },
        )

    return image_format


def _load_rgb_image(image_bytes: bytes) -> Image.Image:
    """Decode to RGB with the camera's rotation applied.

    Phone photos carry their rotation in an EXIF tag rather than in the pixel
    data. Without applying it the model sees a sideways image and the search
    quietly gets worse, which is invisible unless you look for it.
    """
    try:
        with Image.open(BytesIO(image_bytes)) as opened_image:
            upright_image = ImageOps.exif_transpose(opened_image)
            try:
                loaded_image = upright_image.convert("RGB")
                loaded_image.load()
            finally:
                # exif_transpose commonly returns a copy. Close that temporary
                # promptly; only the independent RGB result leaves this function.
                if upright_image is not opened_image:
                    upright_image.close()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(message="Invalid or broken image file") from exc

    return loaded_image
