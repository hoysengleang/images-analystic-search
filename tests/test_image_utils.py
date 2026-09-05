from io import BytesIO

import pytest
from PIL import Image

from app.core.errors import (
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedImageTypeError,
)
from app.utils.image_utils import (
    image_to_rgb,
    normalize_image_extension,
    validate_and_load_image,
    validate_image_extension,
)


def make_image_bytes(
    *,
    image_format: str = "PNG",
    mode: str = "RGBA",
) -> bytes:
    image = Image.new(mode, (2, 2), color=(255, 0, 0, 128))
    buffer = BytesIO()
    save_image = image.convert("RGB") if image_format == "JPEG" else image
    save_image.save(buffer, format=image_format)
    return buffer.getvalue()


def test_valid_png_loads_and_converts_to_rgb() -> None:
    image_bytes = make_image_bytes(image_format="PNG", mode="RGBA")

    validated = validate_and_load_image(
        image_bytes,
        max_size_mb=1,
        extension=".png",
    )

    assert validated.image.mode == "RGB"
    assert validated.format == "PNG"
    assert validated.extension == "png"
    assert validated.width == 2
    assert validated.height == 2


def test_valid_jpeg_loads_with_jpg_extension() -> None:
    image_bytes = make_image_bytes(image_format="JPEG", mode="RGB")

    validated = validate_and_load_image(
        image_bytes,
        max_size_mb=1,
        extension=".jpg",
    )

    assert validated.image.mode == "RGB"
    assert validated.format == "JPEG"
    assert validated.extension == "jpeg"


def test_valid_webp_loads_when_supported() -> None:
    image_bytes = make_image_bytes(image_format="WEBP", mode="RGB")

    validated = validate_and_load_image(
        image_bytes,
        max_size_mb=1,
        extension=".webp",
    )

    assert validated.format == "WEBP"
    assert validated.extension == "webp"


def test_rejects_broken_image_bytes() -> None:
    with pytest.raises(InvalidImageError) as exc_info:
        validate_and_load_image(b"not an image", max_size_mb=1)

    assert exc_info.value.code == "INVALID_IMAGE"


def test_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedImageTypeError) as exc_info:
        validate_image_extension(".gif")

    assert exc_info.value.code == "UNSUPPORTED_IMAGE_TYPE"


def test_rejects_extension_format_mismatch() -> None:
    image_bytes = make_image_bytes(image_format="PNG", mode="RGB")

    with pytest.raises(UnsupportedImageTypeError) as exc_info:
        validate_and_load_image(
            image_bytes,
            max_size_mb=1,
            extension=".jpg",
        )

    assert exc_info.value.code == "UNSUPPORTED_IMAGE_TYPE"


def test_rejects_oversized_image_bytes() -> None:
    with pytest.raises(ImageTooLargeError) as exc_info:
        validate_and_load_image(b"0" * 2 * 1024 * 1024, max_size_mb=1)

    assert exc_info.value.code == "IMAGE_TOO_LARGE"


def test_normalizes_extensions() -> None:
    assert normalize_image_extension(".jpg") == "jpeg"
    assert normalize_image_extension("JPEG") == "jpeg"
    assert normalize_image_extension("/tmp/a.PNG") == "png"


def test_image_to_rgb_returns_rgb_copy() -> None:
    image = Image.new("RGBA", (1, 1), color=(0, 0, 0, 0))

    rgb_image = image_to_rgb(image)

    assert rgb_image.mode == "RGB"
