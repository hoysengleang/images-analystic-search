"""Structured JSON logging with credential redaction.

Section 20 of the specification forbids logging API keys, source credentials,
or image bytes. Redaction happens in the formatter so a careless call site
cannot leak a secret that was passed as a log field.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

REDACTED = "[redacted]"

#: Field names whose values are never written to a log line.
SENSITIVE_FIELDS = frozenset(
    {
        "api_key",
        "authorization",
        "credentials",
        "key_hash",
        "password",
        "presented_key",
        "qdrant_api_key",
        "secret",
        "secret_access_key",
        "token",
        "x-api-key",
    }
)

#: Attributes the standard library puts on every record; not our fields.
_STANDARD_ATTRIBUTES = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


def redact(value: Any, *, key: str = "") -> Any:
    if key.lower() in SENSITIVE_FIELDS:
        return REDACTED
    if isinstance(value, dict):
        return {name: redact(item, key=name) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for name, value in record.__dict__.items():
            if name not in _STANDARD_ATTRIBUTES:
                payload[name] = redact(value, key=name)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Install the root handler. Safe to call more than once."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
