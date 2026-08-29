"""Documentation is checked against the code, so it cannot quietly rot.

Every one of these caught a real gap when it was written. They exist because a
setting that is documented but unread, or an error code a caller can receive
but cannot look up, wastes a developer's afternoon.
"""

import ast
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import errors as errors_module
from app.core.config import Settings
from app.main import app

REPO_ROOT = Path(__file__).resolve().parent.parent
README = (REPO_ROOT / "README.md").read_text()
API_REFERENCE = (REPO_ROOT / "docs" / "api-reference.md").read_text()
ENV_EXAMPLE = (REPO_ROOT / ".env.example").read_text()
APP_SOURCE = "\n".join(path.read_text() for path in (REPO_ROOT / "app").rglob("*.py"))

#: Read by the container command or Compose rather than by Python.
PROCESS_LEVEL_SETTINGS = {"APP_HOST", "APP_PORT", "APP_ENV", "APP_VERSION"}

SETTING_ALIASES = sorted(
    field.alias for field in Settings.model_fields.values() if field.alias
)


def test_env_template_is_a_valid_runtime_configuration(monkeypatch) -> None:
    monkeypatch.delenv("SEARCH_ENGINE", raising=False)
    settings = Settings(_env_file=REPO_ROOT / ".env.example")

    assert settings.search_engine == "qdrant"


@pytest.mark.parametrize("alias", SETTING_ALIASES)
def test_every_setting_is_documented_in_the_readme(alias) -> None:
    assert alias in README, f"{alias} is configurable but absent from the README"


@pytest.mark.parametrize("alias", SETTING_ALIASES)
def test_every_setting_appears_in_the_env_template(alias) -> None:
    assert f"{alias}=" in ENV_EXAMPLE, f"{alias} is missing from .env.example"


@pytest.mark.parametrize("alias", SETTING_ALIASES)
def test_every_setting_actually_does_something(alias) -> None:
    """A setting nobody reads is a promise the service does not keep."""
    if alias in PROCESS_LEVEL_SETTINGS:
        return

    field_name = next(
        name for name, field in Settings.model_fields.items() if field.alias == alias
    )
    assert re.search(rf"\.{field_name}\b", APP_SOURCE), (
        f"{alias} is declared but never read; wire it up or remove it"
    )


def live_paths() -> list:
    return sorted(TestClient(app).get("/openapi.json").json()["paths"])


@pytest.mark.parametrize("path", live_paths())
def test_every_endpoint_is_in_the_api_reference(path) -> None:
    documented = path.replace("{collection_name}", "{name}").replace("{image_id}", "{id}")
    assert documented in API_REFERENCE or path in API_REFERENCE, (
        f"{path} is served but not in docs/api-reference.md"
    )


@pytest.mark.parametrize("path", live_paths())
def test_every_endpoint_is_in_the_readme_table(path) -> None:
    documented = path.replace("{collection_name}", "{name}").replace("{image_id}", "{id}")
    assert documented in README or path in README, (
        f"{path} is served but not in the README endpoint table"
    )


def raised_error_codes() -> set:
    """Every error code the service can put in a response."""
    error_classes = {
        name
        for name in dir(errors_module)
        if isinstance(getattr(errors_module, name), type)
        and issubclass(getattr(errors_module, name), errors_module.OpenVisionSearchError)
    }

    codes = set()
    for path in (REPO_ROOT / "app").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            call = None
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                call = node.exc
            elif isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
                call = node.value
            if call is None:
                continue

            name = (
                call.func.id
                if isinstance(call.func, ast.Name)
                else getattr(call.func, "attr", None)
            )
            if name not in error_classes:
                continue

            explicit = next(
                (
                    keyword.value.value
                    for keyword in call.keywords
                    if keyword.arg == "code" and isinstance(keyword.value, ast.Constant)
                ),
                None,
            )
            codes.add(explicit or getattr(errors_module, name).code)

    return codes


@pytest.mark.parametrize("code", sorted(raised_error_codes()))
def test_every_error_code_a_caller_can_receive_is_documented(code) -> None:
    """A caller who cannot look up a code cannot handle it."""
    # Families are documented by prefix rather than one row each.
    if code.startswith("QDRANT_"):
        assert "`QDRANT_*`" in API_REFERENCE
        return

    assert f"`{code}`" in API_REFERENCE, (
        f"{code} can reach a client but is not in the error reference"
    )


def test_the_error_envelope_is_documented() -> None:
    assert '"error"' in API_REFERENCE
    assert "code" in API_REFERENCE and "details" in API_REFERENCE


def test_retryable_errors_are_called_out() -> None:
    """Clients need to know which failures are worth retrying."""
    assert "retry" in API_REFERENCE.lower()
