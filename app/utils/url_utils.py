from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import urlparse

from app.core.errors import BadRequestError


DEFAULT_MAX_REDIRECTS = 3
MAX_ALLOWED_REDIRECTS = 10


@dataclass(frozen=True)
class SafeURL:
    url: str
    timeout_seconds: int
    max_redirects: int

    def request_options(self) -> dict[str, int | bool]:
        return {
            "timeout": self.timeout_seconds,
            "follow_redirects": True,
            "max_redirects": self.max_redirects,
        }


def validate_safe_url(
    url_value: str,
    *,
    allow_private_urls: bool = False,
    timeout_seconds: int = 10,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
) -> SafeURL:
    if not url_value or not url_value.strip():
        raise BadRequestError(
            message="URL is required",
            code="INVALID_URL",
            details={},
        )

    timeout_seconds = _validate_timeout(timeout_seconds)
    max_redirects = _validate_max_redirects(max_redirects)
    url = url_value.strip()
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise BadRequestError(
            message="URL scheme must be http or https",
            code="UNSUPPORTED_URL_SCHEME",
            details={"scheme": parsed.scheme or None},
        )

    if not parsed.hostname:
        raise BadRequestError(
            message="URL must include a hostname",
            code="INVALID_URL",
            details={"url": url},
        )

    if not allow_private_urls and _is_private_or_internal_host(parsed.hostname):
        raise BadRequestError(
            message="Private or internal URLs are not allowed",
            code="PRIVATE_URL_NOT_ALLOWED",
            details={"host": parsed.hostname},
        )

    return SafeURL(
        url=url,
        timeout_seconds=timeout_seconds,
        max_redirects=max_redirects,
    )


def _validate_timeout(timeout_seconds: int) -> int:
    if timeout_seconds < 1:
        raise BadRequestError(
            message="URL timeout must be at least 1 second",
            code="INVALID_URL_TIMEOUT",
            details={"timeout_seconds": timeout_seconds},
        )
    return timeout_seconds


def _validate_max_redirects(max_redirects: int) -> int:
    if max_redirects < 0 or max_redirects > MAX_ALLOWED_REDIRECTS:
        raise BadRequestError(
            message=f"URL redirects must be between 0 and {MAX_ALLOWED_REDIRECTS}",
            code="INVALID_URL_REDIRECT_LIMIT",
            details={"max_redirects": max_redirects},
        )
    return max_redirects


def _is_private_or_internal_host(hostname: str) -> bool:
    normalized_host = hostname.strip().strip("[]").lower().rstrip(".")

    if normalized_host in {"localhost", "localhost.localdomain"}:
        return True
    if normalized_host.endswith(".localhost"):
        return True

    try:
        host_ip = ip_address(normalized_host)
    except ValueError:
        return False

    return any(
        (
            host_ip.is_private,
            host_ip.is_loopback,
            host_ip.is_link_local,
            host_ip.is_reserved,
            host_ip.is_multicast,
            host_ip.is_unspecified,
        )
    )
