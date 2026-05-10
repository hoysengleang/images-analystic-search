from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from app.core.errors import ConflictError, ResourceNotFoundError
from app.schemas.collection import CollectionModelConfig


DEFAULT_METADATA_PATH = Path("data/collections.json")


@dataclass(frozen=True)
class CollectionMetadata:
    collection_name: str
    embedding_provider: str
    embedding_model: str
    embedding_pretrained: str
    vector_size: int
    distance: str
    created_at: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


class CollectionMetadataService:
    def __init__(self, metadata_path: Path = DEFAULT_METADATA_PATH) -> None:
        self.metadata_path = metadata_path

    def save(
        self,
        *,
        collection_name: str,
        model: CollectionModelConfig,
    ) -> CollectionMetadata:
        normalized_name = self._normalize_collection_name(collection_name)
        collections = self._read_all()

        if normalized_name in collections:
            raise ConflictError(
                message=f"Collection already exists: {normalized_name}",
                code="COLLECTION_ALREADY_EXISTS",
                details={"collection_name": normalized_name},
            )

        metadata = CollectionMetadata(
            collection_name=normalized_name,
            embedding_provider=model.provider,
            embedding_model=model.name,
            embedding_pretrained=model.pretrained,
            vector_size=model.vector_size,
            distance=model.distance,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        collections[normalized_name] = metadata.to_dict()
        self._write_all(collections)
        return metadata

    def get(self, collection_name: str) -> CollectionMetadata:
        normalized_name = self._normalize_collection_name(collection_name)
        collections = self._read_all()

        try:
            return self._metadata_from_dict(collections[normalized_name])
        except KeyError as exc:
            raise ResourceNotFoundError(
                message=f"Collection not found: {normalized_name}",
                code="COLLECTION_NOT_FOUND",
                details={"collection_name": normalized_name},
            ) from exc

    def list(self) -> list[CollectionMetadata]:
        collections = self._read_all()
        return [
            self._metadata_from_dict(collections[name])
            for name in sorted(collections)
        ]

    def delete(self, collection_name: str) -> CollectionMetadata:
        normalized_name = self._normalize_collection_name(collection_name)
        collections = self._read_all()

        try:
            deleted = self._metadata_from_dict(collections.pop(normalized_name))
        except KeyError as exc:
            raise ResourceNotFoundError(
                message=f"Collection not found: {normalized_name}",
                code="COLLECTION_NOT_FOUND",
                details={"collection_name": normalized_name},
            ) from exc

        self._write_all(collections)
        return deleted

    def _read_all(self) -> dict[str, dict[str, str | int]]:
        if not self.metadata_path.exists():
            return {}

        with self.metadata_path.open("r", encoding="utf-8") as metadata_file:
            data = json.load(metadata_file)

        if not isinstance(data, dict):
            return {}

        return data

    def _write_all(self, collections: dict[str, dict[str, str | int]]) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.metadata_path.with_suffix(f"{self.metadata_path.suffix}.tmp")

        with temp_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(collections, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")

        temp_path.replace(self.metadata_path)

    def _metadata_from_dict(self, data: dict[str, str | int]) -> CollectionMetadata:
        return CollectionMetadata(
            collection_name=str(data["collection_name"]),
            embedding_provider=str(data["embedding_provider"]),
            embedding_model=str(data["embedding_model"]),
            embedding_pretrained=str(data["embedding_pretrained"]),
            vector_size=int(data["vector_size"]),
            distance=str(data["distance"]),
            created_at=str(data["created_at"]),
        )

    def _normalize_collection_name(self, collection_name: str) -> str:
        return collection_name.strip()


def get_collection_metadata_service() -> CollectionMetadataService:
    return CollectionMetadataService()
