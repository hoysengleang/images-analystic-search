"""Helpers for multipart form fields that carry structured values."""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import UploadFile

from app.core.errors import BadRequestError, ImageTooLargeError

UPLOAD_CHUNK_SIZE = 64 * 1024


async def read_upload_within_limit(upload: UploadFile, *, max_bytes: int) -> bytes:
    """Read an uploaded file, stopping as soon as it passes the size limit."""
    chunks: list[bytes] = []
    total_bytes = 0

    while True:
        chunk = await upload.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break

        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise ImageTooLargeError(
                details={
                    "filename": upload.filename,
                    "max_size_bytes": max_bytes,
                },
            )
        chunks.append(chunk)

    return b"".join(chunks)


def parse_metadata_form_field(metadata: Optional[str]) -> dict[str, Any]:
    """Parse a JSON object sent as a multipart form field."""
    if metadata is None or not metadata.strip():
        return {}

    try:
        parsed: Any = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise _invalid_metadata_error() from exc

    if not isinstance(parsed, dict):
        raise _invalid_metadata_error()

    return parsed


def _invalid_metadata_error() -> BadRequestError:
    return BadRequestError(
        message="Upload metadata must be a valid JSON object",
        code="INVALID_UPLOAD_METADATA",
    )
