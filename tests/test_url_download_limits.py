from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image

from app.core.config import Settings
from app.core.errors import ImageTooLargeError, UnsupportedImageTypeError
from app.services.image_loader import ImageLoader


# Fake DNS so the tests never touch the network.
def PUBLIC_RESOLVER(hostname: str) -> list:
    return ["93.184.216.34"]


def make_settings() -> Settings:
    return Settings(
        allowed_image_root=Path("/data/images"),
        max_image_size_mb=1,
        url_download_timeout_seconds=3,
    )


def make_loader(handler) -> ImageLoader:
    return ImageLoader(
        host_resolver=PUBLIC_RESOLVER,
        settings=make_settings(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def make_image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_declared_content_length_over_the_limit_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-length": str(5 * 1024 * 1024), "content-type": "image/png"},
            content=make_image_bytes(),
        )

    with pytest.raises(ImageTooLargeError) as exc_info:
        make_loader(handler).load_from_url("https://example.com/big.png")

    assert exc_info.value.code == "IMAGE_TOO_LARGE"


def test_oversized_body_is_refused_mid_stream() -> None:
    over_limit_chunks = (b"x" * 64 * 1024 for _ in range(32))

    def handler(request: httpx.Request) -> httpx.Response:
        # No content-length, so the cap has to hold while streaming.
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=over_limit_chunks,
        )

    with pytest.raises(ImageTooLargeError):
        make_loader(handler).load_from_url("https://example.com/chunked.png")


def test_html_error_page_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=b"<html>404 not found</html>",
        )

    with pytest.raises(UnsupportedImageTypeError) as exc_info:
        make_loader(handler).load_from_url("https://example.com/missing.png")

    assert exc_info.value.code == "UNSUPPORTED_IMAGE_TYPE"


def test_octet_stream_from_object_storage_is_accepted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/octet-stream"},
            content=make_image_bytes(),
        )

    image = make_loader(handler).load_from_url("https://example.com/signed.png")

    assert image.mode == "RGB"
