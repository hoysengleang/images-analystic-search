from __future__ import annotations

import base64
import binascii
from pathlib import Path
from threading import Lock
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from PIL import Image

from app.core.config import Settings
from app.core.errors import (
    BadRequestError,
    ImageTooLargeError,
    InvalidImageError,
    ServiceUnavailableError,
    UnsupportedImageTypeError,
)
from app.schemas.image import ImageSource
from app.utils.image_utils import validate_and_load_image
from app.utils.path_utils import resolve_safe_image_path
from app.utils.url_utils import HostResolver, SafeURL, validate_safe_url

#: Content types an object store may use for an image it does not sniff.
OPAQUE_BINARY_MEDIA_TYPES = frozenset({"application/octet-stream", "binary/octet-stream"})

DOWNLOAD_CHUNK_SIZE = 64 * 1024


class ImageLoader:
    """Turns an :class:`ImageSource` into a validated RGB Pillow image.

    This is the only place that reads bytes from the outside world, so every
    path, URL, and base64 payload passes the same safety checks.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        http_client: Optional[httpx.Client] = None,
        host_resolver: Optional[HostResolver] = None,
    ) -> None:
        self.settings = settings
        self.host_resolver = host_resolver
        self._injected_client = http_client
        self._client: Optional[httpx.Client] = None
        self._client_lock = Lock()

    def load_from_path(self, path_value: str) -> Image.Image:
        image_path = resolve_safe_image_path(
            path_value,
            self.settings.allowed_image_root,
        )
        image_bytes = self._read_path_within_limit(image_path)

        return validate_and_load_image(
            image_bytes,
            max_size_mb=self.settings.max_image_size_mb,
            extension=image_path.suffix,
            allowed_extensions=self.settings.allowed_image_extensions,
            max_pixels=self.settings.max_image_pixels,
        ).image

    def _read_path_within_limit(self, image_path: Path) -> bytes:
        """Read a local image without ever retaining more than the size cap."""
        max_bytes = self.settings.max_image_size_bytes
        chunks: list[bytes] = []
        total_bytes = 0

        try:
            with image_path.open("rb") as image_file:
                while chunk := image_file.read(DOWNLOAD_CHUNK_SIZE):
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise ImageTooLargeError(
                            details={
                                "path": str(image_path),
                                "size_bytes": total_bytes,
                                "max_size_bytes": max_bytes,
                            },
                        )
                    chunks.append(chunk)
        except OSError as exc:
            raise InvalidImageError(
                message="Could not read image file",
                details={"path": str(image_path)},
            ) from exc

        return b"".join(chunks)

    def load_from_url(self, url_value: str) -> Image.Image:
        safe_url = self._validate_url(url_value)
        image_bytes, final_url = self._download(safe_url.url)
        # After redirects the last URL is the one that described the bytes.
        extension = Path(urlparse(final_url).path).suffix or None

        return validate_and_load_image(
            image_bytes,
            max_size_mb=self.settings.max_image_size_mb,
            extension=extension,
            allowed_extensions=self.settings.allowed_image_extensions,
            max_pixels=self.settings.max_image_pixels,
        ).image

    def load_from_base64(self, value: str) -> Image.Image:
        if value.strip().startswith("data:"):
            raise BadRequestError(
                message="Base64 image data must not include a data URL prefix",
                code="INVALID_BASE64_IMAGE",
            )

        try:
            normalized_value = "".join(value.split())
            image_bytes = base64.b64decode(normalized_value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise InvalidImageError(message="Invalid base64 image data") from exc

        return validate_and_load_image(
            image_bytes,
            max_size_mb=self.settings.max_image_size_mb,
            allowed_extensions=self.settings.allowed_image_extensions,
            max_pixels=self.settings.max_image_pixels,
        ).image

    def load_from_source(self, source: ImageSource) -> Image.Image:
        if source.type == "path":
            return self.load_from_path(source.value)
        if source.type == "url":
            return self.load_from_url(source.value)
        if source.type == "base64":
            return self.load_from_base64(source.value)

        raise BadRequestError(
            message="Unsupported image source type",
            code="UNSUPPORTED_IMAGE_SOURCE",
            details={"source_type": source.type},
        )

    def close(self) -> None:
        """Release the pooled HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def _validate_url(self, url_value: str) -> SafeURL:
        return validate_safe_url(
            url_value,
            allow_private_urls=self.settings.allow_private_urls,
            timeout_seconds=self.settings.url_download_timeout_seconds,
            max_redirects=self.settings.max_redirects,
            host_resolver=self.host_resolver,
        )

    def _get_client(self) -> httpx.Client:
        """Return a pooled client so bulk indexing reuses connections."""
        if self._injected_client is not None:
            return self._injected_client

        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = httpx.Client(
                        timeout=self.settings.url_download_timeout_seconds,
                        follow_redirects=False,
                        limits=httpx.Limits(
                            max_connections=32,
                            max_keepalive_connections=16,
                        ),
                    )

        return self._client

    def _download(self, url: str) -> tuple:
        """Download an image, re-checking safety at every redirect hop.

        Redirects are followed by hand because letting the HTTP client follow
        them would apply the URL allow-list only to the first hop, leaving a
        public URL free to bounce the request onto an internal address.
        """
        client = self._get_client()
        current_url = url

        try:
            for _ in range(self.settings.max_redirects + 1):
                with client.stream(
                    "GET",
                    current_url,
                    follow_redirects=False,
                    timeout=self.settings.url_download_timeout_seconds,
                ) as response:
                    if response.is_redirect:
                        current_url = self._next_redirect_url(response, current_url)
                        continue

                    response.raise_for_status()
                    self._validate_content_type(
                        response.headers.get("content-type"),
                        current_url,
                    )
                    self._validate_content_length(
                        response.headers.get("content-length"),
                        current_url,
                    )
                    return self._read_within_limit(response, current_url), current_url
        except httpx.HTTPStatusError as exc:
            raise BadRequestError(
                message="Image URL returned an unsuccessful status",
                code="IMAGE_URL_BAD_STATUS",
                details={"url": current_url, "status_code": exc.response.status_code},
            ) from exc
        except httpx.HTTPError as exc:
            raise ServiceUnavailableError(
                message="Could not download image URL",
                code="IMAGE_URL_DOWNLOAD_FAILED",
                details={"url": current_url, "error": str(exc)},
            ) from exc

        raise BadRequestError(
            message="Image URL exceeded the allowed number of redirects",
            code="TOO_MANY_REDIRECTS",
            details={"url": url, "max_redirects": self.settings.max_redirects},
        )

    def _next_redirect_url(self, response: httpx.Response, current_url: str) -> str:
        location = response.headers.get("location")
        if not location:
            raise BadRequestError(
                message="Image URL redirected without a target",
                code="INVALID_REDIRECT",
                details={"url": current_url},
            )

        # Relative redirects are legal, so resolve against the current URL
        # before the target is validated.
        return self._validate_url(urljoin(current_url, location)).url

    def _read_within_limit(self, response: httpx.Response, url: str) -> bytes:
        max_bytes = self.settings.max_image_size_bytes
        chunks: list = []
        downloaded_bytes = 0

        for chunk in response.iter_bytes(DOWNLOAD_CHUNK_SIZE):
            downloaded_bytes += len(chunk)
            if downloaded_bytes > max_bytes:
                raise ImageTooLargeError(
                    message="Image URL exceeds the maximum image size",
                    details={"url": url, "max_size_bytes": max_bytes},
                )
            chunks.append(chunk)

        return b"".join(chunks)

    def _validate_content_type(self, content_type: Optional[str], url: str) -> None:
        """Reject obviously wrong responses such as HTML error pages.

        Object stores often serve images as ``application/octet-stream``, so an
        unknown binary type is allowed through and left to Pillow to verify.
        """
        if not content_type:
            return

        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type in OPAQUE_BINARY_MEDIA_TYPES:
            return

        if media_type not in self.settings.allowed_image_types:
            raise UnsupportedImageTypeError(
                message="Image URL returned an unsupported content type",
                details={
                    "url": url,
                    "content_type": media_type,
                    "allowed_image_types": self.settings.allowed_image_types,
                },
            )

    def _validate_content_length(self, content_length: Optional[str], url: str) -> None:
        if not content_length:
            return

        try:
            declared_bytes = int(content_length)
        except ValueError:
            return

        if declared_bytes > self.settings.max_image_size_bytes:
            raise ImageTooLargeError(
                message="Image URL exceeds the maximum image size",
                details={
                    "url": url,
                    "size_bytes": declared_bytes,
                    "max_size_bytes": self.settings.max_image_size_bytes,
                },
            )
