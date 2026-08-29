import base64
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image

from app.core.config import Settings
from app.core.errors import BadRequestError, ImageTooLargeError, InvalidImageError
from app.schemas.image import ImageSource
from app.services.image_loader import ImageLoader


# Fake DNS so the tests never touch the network.
def PUBLIC_RESOLVER(hostname: str) -> list:
    return ["93.184.216.34"]


def make_image_bytes() -> bytes:
    image = Image.new("RGBA", (2, 2), color=(255, 0, 0, 128))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_settings(allowed_root: Path, *, allow_private_urls: bool = False) -> Settings:
    return Settings(
        allowed_image_root=allowed_root,
        max_image_size_mb=1,
        allow_private_urls=allow_private_urls,
        url_download_timeout_seconds=3,
    )


def test_load_from_path_allowed_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "images"
    allowed_root.mkdir()
    image_path = allowed_root / "a.png"
    image_path.write_bytes(make_image_bytes())
    loader = ImageLoader(
        host_resolver=PUBLIC_RESOLVER, settings=make_settings(allowed_root)
    )

    image = loader.load_from_path(str(image_path))

    assert image.mode == "RGB"
    assert image.size == (2, 2)


def test_load_from_path_blocks_unsafe_path(tmp_path: Path) -> None:
    allowed_root = tmp_path / "images"
    allowed_root.mkdir()
    loader = ImageLoader(
        host_resolver=PUBLIC_RESOLVER, settings=make_settings(allowed_root)
    )

    with pytest.raises(BadRequestError) as exc_info:
        loader.load_from_path("../../secret.png")

    assert exc_info.value.code == "IMAGE_PATH_NOT_ALLOWED"


def test_load_from_path_stops_at_size_limit(tmp_path: Path) -> None:
    allowed_root = tmp_path / "images"
    allowed_root.mkdir()
    oversized_path = allowed_root / "oversized.png"
    oversized_path.write_bytes(b"x" * (1024 * 1024 + 1))
    loader = ImageLoader(
        host_resolver=PUBLIC_RESOLVER,
        settings=make_settings(allowed_root),
    )

    with pytest.raises(ImageTooLargeError) as exc_info:
        loader.load_from_path(str(oversized_path))

    assert exc_info.value.details["max_size_bytes"] == 1024 * 1024


def test_load_from_url() -> None:
    image_bytes = make_image_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/a.png"
        return httpx.Response(200, content=image_bytes)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    loader = ImageLoader(
        host_resolver=PUBLIC_RESOLVER,
        settings=make_settings(Path("/data/images")),
        http_client=http_client,
    )

    image = loader.load_from_url("https://example.com/a.png")

    assert image.mode == "RGB"
    assert image.size == (2, 2)
    http_client.close()


def test_load_from_url_blocks_private_url_by_default() -> None:
    loader = ImageLoader(
        host_resolver=PUBLIC_RESOLVER, settings=make_settings(Path("/data/images"))
    )

    with pytest.raises(BadRequestError) as exc_info:
        loader.load_from_url("http://127.0.0.1/a.png")

    assert exc_info.value.code == "PRIVATE_URL_NOT_ALLOWED"


def test_load_from_base64() -> None:
    encoded = base64.b64encode(make_image_bytes()).decode("ascii")
    loader = ImageLoader(
        host_resolver=PUBLIC_RESOLVER, settings=make_settings(Path("/data/images"))
    )

    image = loader.load_from_base64(encoded)

    assert image.mode == "RGB"
    assert image.size == (2, 2)


def test_load_from_source_dispatches_base64() -> None:
    encoded = base64.b64encode(make_image_bytes()).decode("ascii")
    source = ImageSource(type="base64", value=encoded)
    loader = ImageLoader(
        host_resolver=PUBLIC_RESOLVER, settings=make_settings(Path("/data/images"))
    )

    image = loader.load_from_source(source)

    assert image.mode == "RGB"


def test_broken_image_is_rejected_from_base64() -> None:
    encoded = base64.b64encode(b"not an image").decode("ascii")
    loader = ImageLoader(
        host_resolver=PUBLIC_RESOLVER, settings=make_settings(Path("/data/images"))
    )

    with pytest.raises(InvalidImageError) as exc_info:
        loader.load_from_base64(encoded)

    assert exc_info.value.code == "INVALID_IMAGE"


def test_broken_image_is_rejected_from_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not an image")

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    loader = ImageLoader(
        host_resolver=PUBLIC_RESOLVER,
        settings=make_settings(Path("/data/images")),
        http_client=http_client,
    )

    with pytest.raises(InvalidImageError) as exc_info:
        loader.load_from_url("https://example.com/a.png")

    assert exc_info.value.code == "INVALID_IMAGE"
    http_client.close()
