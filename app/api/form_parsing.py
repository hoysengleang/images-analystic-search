"""Helpers for multipart form fields that carry structured values."""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import UploadFile
from pydantic import ValidationError

from app.core.errors import BadRequestError, ImageTooLargeError
from app.schemas.image import CropRectangle

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


def parse_crop_form_fields(
    *,
    crop_x: Optional[float],
    crop_y: Optional[float],
    crop_width: Optional[float],
    crop_height: Optional[float],
) -> Optional[CropRectangle]:
    values = (crop_x, crop_y, crop_width, crop_height)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise BadRequestError(
            message=(
                "A crop needs all four of crop_x, crop_y, crop_width and crop_height"
            ),
            code="INCOMPLETE_CROP",
            details={
                "crop_x": crop_x,
                "crop_y": crop_y,
                "crop_width": crop_width,
                "crop_height": crop_height,
            },
        )

    try:
        return CropRectangle(
            x=crop_x,
            y=crop_y,
            width=crop_width,
            height=crop_height,
        )
    except ValidationError as exc:
        raise BadRequestError(
            message="Crop rectangle is not a valid region of the image",
            code="INVALID_CROP",
            details={"errors": [error["msg"] for error in exc.errors()]},
        ) from exc
