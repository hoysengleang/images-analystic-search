from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

from app.core.errors import ImageTooLargeError, InvalidImageError, UnsupportedImageTypeError


ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


@dataclass(frozen=True)
class ValidatedImage:
    image: Image.Image
    format: str
    extension: str
    size_bytes: int
    width: int
    height: int


def normalize_image_extension(extension_or_path: str) -> str:
    suffix = Path(extension_or_path).suffix
    extension = suffix if suffix else extension_or_path
    normalized = extension.strip().lower().lstrip(".")

    if normalized == "jpg":
        return "jpeg"
    return normalized


def validate_image_extension(extension_or_path: str) -> str:
    extension = normalize_image_extension(extension_or_path)

    if extension not in {"jpeg", "png", "webp"}:
        raise UnsupportedImageTypeError(
            message="Unsupported image file extension",
            details={"extension": extension_or_path},
        )

    return extension


def validate_and_load_image(
    image_bytes: bytes,
    *,
    max_size_mb: int,
    extension: Optional[str] = None,
) -> ValidatedImage:
    _validate_image_size(image_bytes, max_size_mb=max_size_mb)

    if extension is not None:
        validate_image_extension(extension)

    try:
        with Image.open(BytesIO(image_bytes)) as opened_image:
            opened_image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            message="Invalid or broken image file",
            details={},
        ) from exc

    try:
        with Image.open(BytesIO(image_bytes)) as opened_image:
            image_format = (opened_image.format or "").upper()
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise UnsupportedImageTypeError(
                    message="Unsupported image format",
                    details={"format": image_format or None},
                )

            loaded_image = opened_image.convert("RGB")
            loaded_image.load()
    except UnsupportedImageTypeError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            message="Invalid or broken image file",
            details={},
        ) from exc

    image_extension = _extension_from_format(image_format)
    if extension is not None:
        expected_extension = validate_image_extension(extension)
        if expected_extension != image_extension:
            raise UnsupportedImageTypeError(
                message="Image file extension does not match image format",
                details={
                    "extension": expected_extension,
                    "format": image_extension,
                },
            )

    return ValidatedImage(
        image=loaded_image,
        format=image_format,
        extension=image_extension,
        size_bytes=len(image_bytes),
        width=loaded_image.width,
        height=loaded_image.height,
    )


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


def _extension_from_format(image_format: str) -> str:
    if image_format == "JPEG":
        return "jpeg"
    return image_format.lower()
