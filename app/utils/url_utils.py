from __future__ import annotations

import socket
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Callable, Optional
from urllib.parse import urlparse

from app.core.errors import BadRequestError

#: Resolves a hostname to the IP addresses a connection would actually use.
HostResolver = Callable[[str], list]

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


def default_host_resolver(hostname: str) -> list:
    """Return every IP address ``hostname`` currently resolves to."""
    address_infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    return [address_info[4][0] for address_info in address_infos]


def validate_safe_url(
    url_value: str,
    *,
    allow_private_urls: bool = False,
    timeout_seconds: int = 10,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    host_resolver: Optional[HostResolver] = None,
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

    if not allow_private_urls:
        _reject_private_hostname(parsed.hostname)
        # A public-looking name can still resolve to an internal address, so
        # check what a connection would actually reach.
        _reject_private_addresses(
            parsed.hostname,
            host_resolver or default_host_resolver,
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


def _reject_private_hostname(hostname: str) -> None:
    """Reject loopback names and IP literals that are already internal."""
    normalized_host = hostname.strip().strip("[]").lower().rstrip(".")

    if normalized_host in {"localhost", "localhost.localdomain"} or (
        normalized_host.endswith(".localhost")
    ):
        raise _private_url_error(hostname)

    try:
        host_ip = ip_address(normalized_host)
    except ValueError:
        return

    if _is_internal_address(host_ip):
        raise _private_url_error(hostname)


def _reject_private_addresses(hostname: str, host_resolver: HostResolver) -> None:
    """Reject hostnames that resolve to an internal address.

    This is what stops a public name whose DNS record points at 127.0.0.1 or
    at a cloud metadata endpoint. A name that resolves differently between this
    check and the connection itself can still slip through; pin egress at the
    network level if you need a guarantee.
    """
    try:
        addresses = host_resolver(hostname)
    except Exception as exc:
        raise BadRequestError(
            message="Could not resolve the URL hostname",
            code="URL_HOST_RESOLUTION_FAILED",
            details={"host": hostname},
        ) from exc

    if not addresses:
        raise BadRequestError(
            message="Could not resolve the URL hostname",
            code="URL_HOST_RESOLUTION_FAILED",
            details={"host": hostname},
        )

    for address in addresses:
        try:
            resolved_ip = ip_address(address)
        except ValueError:
            continue

        if _is_internal_address(resolved_ip):
            raise _private_url_error(hostname, resolved_address=address)


def _is_internal_address(host_ip) -> bool:
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


def _private_url_error(
    hostname: str,
    *,
    resolved_address: Optional[str] = None,
) -> BadRequestError:
    details = {"host": hostname}
    if resolved_address is not None:
        details["resolved_address"] = resolved_address

    return BadRequestError(
        message="Private or internal URLs are not allowed",
        code="PRIVATE_URL_NOT_ALLOWED",
        details=details,
    )
