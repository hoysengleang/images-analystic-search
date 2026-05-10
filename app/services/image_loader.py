from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from PIL import Image

from app.core.config import Settings, get_settings
from app.core.errors import BadRequestError, InvalidImageError, ServiceUnavailableError
from app.schemas.image import ImageSource
from app.utils.image_utils import validate_and_load_image
from app.utils.path_utils import resolve_safe_image_path
from app.utils.url_utils import DEFAULT_MAX_REDIRECTS, validate_safe_url


class ImageLoader:
    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        http_client: Optional[httpx.Client] = None,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
    ) -> None:
        self.settings = settings or get_settings()
        self.http_client = http_client
        self.max_redirects = max_redirects

    def load_from_path(self, path_value: str) -> Image.Image:
        image_path = resolve_safe_image_path(
            path_value,
            self.settings.allowed_image_root,
        )

        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            raise InvalidImageError(
                message="Could not read image file",
                details={"path": str(image_path)},
            ) from exc

        return validate_and_load_image(
            image_bytes,
            max_size_mb=self.settings.max_image_size_mb,
            extension=image_path.suffix,
        ).image

    def load_from_url(self, url_value: str) -> Image.Image:
        safe_url = validate_safe_url(
            url_value,
            allow_private_urls=self.settings.allow_private_urls,
            timeout_seconds=self.settings.url_download_timeout_seconds,
            max_redirects=self.max_redirects,
        )
        image_bytes = self._download_url(safe_url.url)
        extension = Path(urlparse(safe_url.url).path).suffix or None

        return validate_and_load_image(
            image_bytes,
            max_size_mb=self.settings.max_image_size_mb,
            extension=extension,
        ).image

    def load_from_base64(self, value: str) -> Image.Image:
        if value.strip().startswith("data:"):
            raise BadRequestError(
                message="Base64 image data must not include a data URL prefix",
                code="INVALID_BASE64_IMAGE",
                details={},
            )

        try:
            normalized_value = "".join(value.split())
            image_bytes = base64.b64decode(normalized_value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise InvalidImageError(
                message="Invalid base64 image data",
                details={},
            ) from exc

        return validate_and_load_image(
            image_bytes,
            max_size_mb=self.settings.max_image_size_mb,
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

    def _download_url(self, url: str) -> bytes:
        try:
            if self.http_client is not None:
                response = self.http_client.get(
                    url,
                    follow_redirects=True,
                    timeout=self.settings.url_download_timeout_seconds,
                )
            else:
                with httpx.Client(
                    follow_redirects=True,
                    timeout=self.settings.url_download_timeout_seconds,
                    max_redirects=self.max_redirects,
                ) as client:
                    response = client.get(url)

            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise BadRequestError(
                message="Image URL returned an unsuccessful status",
                code="IMAGE_URL_BAD_STATUS",
                details={
                    "url": url,
                    "status_code": exc.response.status_code,
                },
            ) from exc
        except httpx.HTTPError as exc:
            raise ServiceUnavailableError(
                message="Could not download image URL",
                code="IMAGE_URL_DOWNLOAD_FAILED",
                details={"url": url, "error": str(exc)},
            ) from exc

        return response.content


def get_image_loader() -> ImageLoader:
    return ImageLoader()
