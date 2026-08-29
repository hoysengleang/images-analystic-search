"""Shared constants for image handling.

Kept in one place so the image utilities, the settings, and the loaders all
agree on which formats the service accepts.
"""

from __future__ import annotations

from typing import Final

#: Supported upload MIME types mapped to their canonical file extension.
MIME_TYPE_TO_EXTENSION: Final[dict[str, str]] = {
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/png": "png",
    "image/webp": "webp",
}

#: File extension aliases normalized to a single canonical extension.
EXTENSION_ALIASES: Final[dict[str, str]] = {"jpg": "jpeg"}

#: Pillow format names mapped to their canonical file extension.
PIL_FORMAT_TO_EXTENSION: Final[dict[str, str]] = {
    "JPEG": "jpeg",
    "PNG": "png",
    "WEBP": "webp",
}

#: Every extension the service can decode, regardless of configuration.
SUPPORTED_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    PIL_FORMAT_TO_EXTENSION.values()
)
