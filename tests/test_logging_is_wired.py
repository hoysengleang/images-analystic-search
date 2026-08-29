"""Structured logging has to be switched on, not merely available.

It was written once and left unwired, so the application emitted plain
uvicorn logs and the redaction never ran.
"""

import json
import logging

from app.core.config import Settings
from app.core.logging import JsonFormatter
from app.main import create_app


def test_creating_the_app_installs_the_json_formatter() -> None:
    logging.getLogger().handlers.clear()

    create_app(Settings(log_level="INFO", log_json=True))

    handlers = logging.getLogger().handlers
    assert handlers, "create_app installed no log handler"
    assert any(isinstance(h.formatter, JsonFormatter) for h in handlers)


def test_the_configured_level_is_applied() -> None:
    create_app(Settings(log_level="WARNING"))

    assert logging.getLogger().level == logging.WARNING


def test_plain_text_logging_can_be_chosen() -> None:
    create_app(Settings(log_json=False))

    handler = logging.getLogger().handlers[0]
    assert not isinstance(handler.formatter, JsonFormatter)


def test_credentials_logged_by_the_running_app_are_redacted(capsys) -> None:
    create_app(Settings(log_level="INFO", log_json=True))

    logging.getLogger("app.test").info(
        "indexed a batch", extra={"tenant_id": "t1", "api_key": "ovs_supersecret"}
    )

    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["tenant_id"] == "t1"
    assert line["api_key"] == "[redacted]"
    assert "ovs_supersecret" not in json.dumps(line)
