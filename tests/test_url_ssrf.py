"""The URL allow-list has to hold at every hop, not just the first one."""

from pathlib import Path

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import BadRequestError
from app.services.image_loader import ImageLoader

CLOUD_METADATA = "http://169.254.169.254/latest/meta-data/"
PUBLIC_ADDRESS = "93.184.216.34"


def resolver_for(mapping: dict):
    return lambda hostname: [mapping.get(hostname, PUBLIC_ADDRESS)]


def make_loader(handler, *, mapping=None, max_redirects=3) -> ImageLoader:
    return ImageLoader(
        settings=Settings(
            allowed_image_root=Path("/data/images"),
            max_image_size_mb=1,
            max_redirects=max_redirects,
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        host_resolver=resolver_for(mapping or {}),
    )


def test_public_host_resolving_to_an_internal_address_is_refused() -> None:
    contacted: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        contacted.append(str(request.url))
        return httpx.Response(200, content=b"secrets")

    loader = make_loader(handler, mapping={"evil.example.com": "169.254.169.254"})

    with pytest.raises(BadRequestError) as exc_info:
        loader.load_from_url("https://evil.example.com/a.png")

    assert exc_info.value.code == "PRIVATE_URL_NOT_ALLOWED"
    assert contacted == [], "no request should have been made"


def test_redirect_to_an_internal_address_is_refused() -> None:
    contacted: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        contacted.append(str(request.url))
        if request.url.host == "cdn.example.com":
            return httpx.Response(302, headers={"location": CLOUD_METADATA})
        return httpx.Response(200, content=b"secrets")

    loader = make_loader(handler, mapping={"169.254.169.254": "169.254.169.254"})

    with pytest.raises(BadRequestError) as exc_info:
        loader.load_from_url("https://cdn.example.com/a.png")

    assert exc_info.value.code == "PRIVATE_URL_NOT_ALLOWED"
    assert CLOUD_METADATA not in contacted
    assert all("169.254.169.254" not in url for url in contacted)


def test_redirect_chain_longer_than_the_limit_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://cdn.example.com/next"})

    loader = make_loader(handler, max_redirects=2)

    with pytest.raises(BadRequestError) as exc_info:
        loader.load_from_url("https://cdn.example.com/a.png")

    assert exc_info.value.code == "TOO_MANY_REDIRECTS"


def test_a_redirect_to_a_public_host_still_works() -> None:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (2, 2), (0, 200, 0)).save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a.png":
            return httpx.Response(
                302,
                headers={"location": "https://images.example.com/real.png"},
            )
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=image_bytes,
        )

    image = make_loader(handler).load_from_url("https://cdn.example.com/a.png")

    assert image.mode == "RGB"
    assert image.size == (2, 2)


def test_relative_redirects_are_resolved_and_checked() -> None:
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/a.png":
            return httpx.Response(302, headers={"location": "/moved/b.png"})
        return httpx.Response(404)

    loader = make_loader(handler)

    with pytest.raises(BadRequestError):
        loader.load_from_url("https://cdn.example.com/a.png")

    assert seen[1] == "https://cdn.example.com/moved/b.png"


def test_unresolvable_host_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"never reached")

    def failing_resolver(hostname: str):
        raise OSError("NXDOMAIN")

    loader = ImageLoader(
        settings=Settings(allowed_image_root=Path("/data/images")),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        host_resolver=failing_resolver,
    )

    with pytest.raises(BadRequestError) as exc_info:
        loader.load_from_url("https://nope.example.com/a.png")

    assert exc_info.value.code == "URL_HOST_RESOLUTION_FAILED"


def test_dns_checks_are_skipped_when_private_urls_are_allowed() -> None:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (2, 2), (10, 10, 10)).save(buffer, format="PNG")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=buffer.getvalue(),
        )

    loader = ImageLoader(
        settings=Settings(
            allowed_image_root=Path("/data/images"),
            allow_private_urls=True,
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        host_resolver=lambda hostname: ["10.0.0.5"],
    )

    assert loader.load_from_url("http://internal-cdn/a.png").mode == "RGB"
