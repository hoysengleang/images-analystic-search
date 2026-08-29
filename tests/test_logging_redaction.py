"""Section 20 forbids logging credentials or image bytes."""

import json
import logging

from app.core.logging import JsonFormatter, redact


def format_record(**fields) -> dict:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="event happened",
        args=(),
        exc_info=None,
    )
    for name, value in fields.items():
        setattr(record, name, value)
    return json.loads(JsonFormatter().format(record))


def test_log_lines_are_json_with_the_expected_envelope() -> None:
    payload = format_record(tenant_id="t1")

    assert payload["message"] == "event happened"
    assert payload["level"] == "INFO"
    assert payload["tenant_id"] == "t1"
    assert "timestamp" in payload


def test_credentials_are_redacted() -> None:
    payload = format_record(api_key="ovs_secret", token="abc", tenant_id="t1")

    assert payload["api_key"] == "[redacted]"
    assert payload["token"] == "[redacted]"
    assert payload["tenant_id"] == "t1"


def test_redaction_reaches_nested_values() -> None:
    payload = format_record(config={"bucket": "products", "secret": "shhh"})

    assert payload["config"] == {"bucket": "products", "secret": "[redacted]"}


def test_image_bytes_are_summarised_not_written() -> None:
    payload = format_record(image=b"\xff" * 2048)

    assert payload["image"] == "<2048 bytes>"


def test_redaction_is_case_insensitive() -> None:
    assert redact("value", key="Authorization") == "[redacted]"
    assert redact("value", key="X-API-Key") == "[redacted]"
