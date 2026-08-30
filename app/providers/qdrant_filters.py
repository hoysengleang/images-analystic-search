"""Compile public metadata filters into Qdrant filter objects."""

from __future__ import annotations

import re
from typing import Any, Optional

from qdrant_client.http import models as qdrant_models

from app.core.errors import BadRequestError

#: Bounds accepted by a range filter, matching Qdrant's operator names.
RANGE_OPERATORS = frozenset({"gt", "gte", "lt", "lte"})

#: Metadata keys Qdrant can safely address as a payload path.
FILTER_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*(\.[A-Za-z0-9_-]+)*$")


def build_qdrant_filter(
    filters: dict[str, Any],
) -> Optional[qdrant_models.Filter]:
    """Return a validated Qdrant filter or ``None`` for no constraints."""
    if not filters:
        return None

    conditions = [
        qdrant_models.FieldCondition(
            key=f"metadata.{_validated_key(key)}",
            **_condition_for(key, value),
        )
        for key, value in filters.items()
    ]
    return qdrant_models.Filter(must=conditions)


def _validated_key(key: str) -> str:
    if not FILTER_KEY_PATTERN.match(key):
        raise _unsupported_filter(
            key,
            None,
            "a filter field must be letters, digits, underscore, or hyphen, "
            "optionally dotted for nested fields",
        )
    return key


def _condition_for(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, (bool, int, float, str)):
        return {"match": qdrant_models.MatchValue(value=value)}

    if isinstance(value, (list, tuple, set)):
        return {"match": qdrant_models.MatchAny(any=_match_any(key, value))}

    if isinstance(value, dict):
        return {"range": qdrant_models.Range(**_range_bounds(key, value))}

    raise _unsupported_filter(
        key,
        value,
        "a filter value must be a scalar, a homogeneous list, or a numeric range",
    )


def _match_any(key: str, value: Any) -> list[str] | list[int]:
    values = list(value)
    all_strings = values and all(isinstance(item, str) for item in values)
    all_integers = values and all(
        isinstance(item, int) and not isinstance(item, bool) for item in values
    )

    if not (all_strings or all_integers):
        raise _unsupported_filter(
            key,
            value,
            "a list filter must hold only strings or only integers",
        )

    return values


def _range_bounds(key: str, value: dict[str, Any]) -> dict[str, Any]:
    unknown = set(value) - RANGE_OPERATORS
    if unknown or not value:
        raise _unsupported_filter(
            key,
            value,
            f"range filters accept only {', '.join(sorted(RANGE_OPERATORS))}",
        )

    if not all(
        isinstance(bound, (int, float)) and not isinstance(bound, bool)
        for bound in value.values()
    ):
        raise _unsupported_filter(key, value, "range bounds must be numbers")

    return dict(value)


def _unsupported_filter(
    key: str,
    value: Any,
    reason: str,
) -> BadRequestError:
    return BadRequestError(
        message=f"Unsupported filter for '{key}': {reason}",
        code="UNSUPPORTED_FILTER",
        details={"field": key, "value": value},
    )
