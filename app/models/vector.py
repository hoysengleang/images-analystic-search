from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class ModelVersion:
    """Identity of the model that produced a vector.

    Two vectors are only comparable when their model versions are identical,
    so this string is part of every write and every query.
    """

    provider: str
    model: str
    revision: str

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}@{self.revision}"

    @classmethod
    def parse(cls, value: str) -> ModelVersion:
        try:
            provider, remainder = value.split(":", 1)
            model, revision = remainder.rsplit("@", 1)
        except ValueError as exc:
            raise ValueError(
                f"Model version must look like provider:model@revision, got {value!r}"
            ) from exc
        return cls(provider=provider, model=model, revision=revision)


@dataclass(frozen=True)
class VectorRecord:
    id: str
    tenant_id: str
    product_id: str
    image_id: str
    model_version: str
    dimension: int
    vector: list
    normalized: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if len(self.vector) != self.dimension:
            raise ValueError(
                f"Vector length {len(self.vector)} does not match "
                f"declared dimension {self.dimension}"
            )
