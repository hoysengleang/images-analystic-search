import pytest

from app.core.errors import BadRequestError
from app.utils.url_utils import validate_safe_url


def test_allows_normal_https_image_url() -> None:
    safe_url = validate_safe_url("https://example.com/images/a.jpg")

    assert safe_url.url == "https://example.com/images/a.jpg"
    assert safe_url.timeout_seconds == 10
    assert safe_url.max_redirects == 3


def test_allows_normal_http_image_url() -> None:
    safe_url = validate_safe_url("http://example.com/images/a.jpg")

    assert safe_url.url == "http://example.com/images/a.jpg"


def test_blocks_file_url() -> None:
    with pytest.raises(BadRequestError) as exc_info:
        validate_safe_url("file:///etc/passwd")

    assert exc_info.value.code == "UNSUPPORTED_URL_SCHEME"


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/a.jpg",
        "http://127.0.0.1/a.jpg",
        "http://10.0.0.5/a.jpg",
        "http://172.16.0.5/a.jpg",
        "http://192.168.1.20/a.jpg",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/a.jpg",
    ],
)
def test_blocks_private_or_internal_urls_by_default(url: str) -> None:
    with pytest.raises(BadRequestError) as exc_info:
        validate_safe_url(url)

    assert exc_info.value.code == "PRIVATE_URL_NOT_ALLOWED"


def test_allows_private_url_when_enabled() -> None:
    safe_url = validate_safe_url(
        "http://127.0.0.1/a.jpg",
        allow_private_urls=True,
    )

    assert safe_url.url == "http://127.0.0.1/a.jpg"


def test_exposes_timeout_and_redirect_options() -> None:
    safe_url = validate_safe_url(
        "https://example.com/a.jpg",
        timeout_seconds=5,
        max_redirects=2,
    )

    assert safe_url.request_options() == {
        "timeout": 5,
        "follow_redirects": True,
        "max_redirects": 2,
    }


def test_rejects_invalid_timeout() -> None:
    with pytest.raises(BadRequestError) as exc_info:
        validate_safe_url("https://example.com/a.jpg", timeout_seconds=0)

    assert exc_info.value.code == "INVALID_URL_TIMEOUT"


def test_rejects_excessive_redirect_limit() -> None:
    with pytest.raises(BadRequestError) as exc_info:
        validate_safe_url("https://example.com/a.jpg", max_redirects=11)

    assert exc_info.value.code == "INVALID_URL_REDIRECT_LIMIT"
