"""Conversions between domain objects and SQLite rows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional


def to_iso(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else value.astimezone(timezone.utc).isoformat()


def from_iso(value: Optional[str]) -> Optional[datetime]:
    return None if value is None else datetime.fromisoformat(value)


def to_json(value: Optional[dict]) -> str:
    return json.dumps(value or {}, sort_keys=True)


def from_json(value: Optional[str]) -> dict:
    if not value:
        return {}
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}


def to_bool(value: Optional[bool]) -> Optional[int]:
    return None if value is None else int(value)


def from_bool(value) -> Optional[bool]:
    return None if value is None else bool(value)
