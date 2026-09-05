"""Preprocessing rules that affect accuracy and safety, not just correctness."""

from io import BytesIO

import pytest
from PIL import Image

from app.core.errors import ImageTooLargeError
from app.utils.image_utils import validate_and_load_image


def jpeg_with_orientation(orientation: int) -> bytes:
    """A wide image tagged as needing a 90-degree rotation."""
    image = Image.new("RGB", (40, 20), color=(255, 0, 0))
    exif = image.getexif()
    exif[274] = orientation  # 274 is the EXIF Orientation tag
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def test_exif_rotation_is_applied_before_embedding() -> None:
    """A phone photo tagged sideways must be uprighted, or the model sees it wrong."""
    validated = validate_and_load_image(
        jpeg_with_orientation(6),
        max_size_mb=5,
        extension=".jpg",
    )

    # Orientation 6 means "rotate 90 degrees", so 40x20 becomes 20x40.
    assert (validated.width, validated.height) == (20, 40)


def test_image_without_orientation_is_untouched() -> None:
    validated = validate_and_load_image(
        jpeg_with_orientation(1),
        max_size_mb=5,
        extension=".jpg",
    )

    assert (validated.width, validated.height) == (40, 20)


def test_pixel_count_ceiling_is_enforced() -> None:
    buffer = BytesIO()
    Image.new("RGB", (500, 500), color=(0, 0, 255)).save(buffer, format="PNG")

    with pytest.raises(ImageTooLargeError) as exc_info:
        validate_and_load_image(
            buffer.getvalue(),
            max_size_mb=5,
            extension=".png",
            max_pixels=100_000,
        )

    assert exc_info.value.code == "IMAGE_TOO_LARGE"
    assert exc_info.value.details["pixels"] == 250_000


def test_a_small_file_that_decodes_huge_is_refused_before_decoding() -> None:
    """A decompression bomb is a tiny file that expands into an enormous bitmap."""
    buffer = BytesIO()
    Image.new("RGB", (4000, 4000), color=(0, 0, 0)).save(buffer, format="PNG")
    bomb = buffer.getvalue()

    assert len(bomb) < 200_000, "flat colour compresses to a tiny file"

    with pytest.raises(ImageTooLargeError):
        validate_and_load_image(
            bomb,
            max_size_mb=5,
            extension=".png",
            max_pixels=1_000_000,
        )


def test_pixel_ceiling_is_optional() -> None:
    buffer = BytesIO()
    Image.new("RGB", (500, 500), color=(0, 255, 0)).save(buffer, format="PNG")

    validated = validate_and_load_image(buffer.getvalue(), max_size_mb=5)

    assert (validated.width, validated.height) == (500, 500)
