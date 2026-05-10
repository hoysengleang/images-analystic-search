from pathlib import Path

import pytest

from app.core.errors import BadRequestError
from app.utils.path_utils import resolve_safe_image_path


def test_allows_absolute_path_inside_allowed_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "data" / "images"
    allowed_root.mkdir(parents=True)
    image_path = allowed_root / "a.jpg"
    image_path.write_bytes(b"image")

    resolved = resolve_safe_image_path(str(image_path), allowed_root)

    assert resolved == image_path.resolve()


def test_allows_data_images_path_when_root_is_data_images() -> None:
    resolved = resolve_safe_image_path(
        "/data/images/a.jpg",
        Path("/data/images"),
    )

    assert resolved == Path("/data/images/a.jpg")


def test_allows_relative_path_inside_allowed_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "images"
    allowed_root.mkdir()
    image_path = allowed_root / "nested" / "a.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"image")

    resolved = resolve_safe_image_path("nested/a.jpg", allowed_root)

    assert resolved == image_path.resolve()


def test_blocks_parent_directory_traversal(tmp_path: Path) -> None:
    allowed_root = tmp_path / "images"
    allowed_root.mkdir()

    with pytest.raises(BadRequestError) as exc_info:
        resolve_safe_image_path("../../secret.txt", allowed_root)

    assert exc_info.value.code == "IMAGE_PATH_NOT_ALLOWED"


def test_blocks_absolute_path_outside_allowed_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "images"
    allowed_root.mkdir()
    outside_path = tmp_path / "outside.jpg"
    outside_path.write_bytes(b"image")

    with pytest.raises(BadRequestError) as exc_info:
        resolve_safe_image_path(str(outside_path), allowed_root)

    assert exc_info.value.code == "IMAGE_PATH_NOT_ALLOWED"


def test_blocks_system_path() -> None:
    with pytest.raises(BadRequestError) as exc_info:
        resolve_safe_image_path("/etc/passwd", Path("/data/images"))

    assert exc_info.value.code == "IMAGE_PATH_NOT_ALLOWED"


def test_blocks_symlink_escape_when_symlink_supported(tmp_path: Path) -> None:
    allowed_root = tmp_path / "images"
    allowed_root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "secret.jpg"
    outside_file.write_bytes(b"secret")
    symlink_path = allowed_root / "linked-secret.jpg"

    try:
        symlink_path.symlink_to(outside_file)
    except OSError:
        pytest.skip("Symlinks are not supported in this environment")

    with pytest.raises(BadRequestError) as exc_info:
        resolve_safe_image_path(str(symlink_path), allowed_root)

    assert exc_info.value.code == "IMAGE_PATH_NOT_ALLOWED"
