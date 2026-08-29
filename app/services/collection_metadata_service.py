from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Optional

from app.core.errors import ConflictError, ResourceNotFoundError
from app.schemas.collection import CollectionModelConfig

try:  # pragma: no cover - platform dependent
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None

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
    framing: str = "pad"
    views: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


class CollectionMetadataService:
    """Durable registry of which embedding model backs each collection.

    Vectors from different models cannot be compared, so the model a collection
    was created with has to survive restarts. Reads are cached against the
    file's mtime, and writes take an exclusive lock, because a lost update here
    orphans a whole collection's vectors.
    """

    def __init__(self, metadata_path: Path = DEFAULT_METADATA_PATH) -> None:
        self.metadata_path = metadata_path
        self._thread_lock = RLock()
        self._cache: Optional[dict] = None
        self._cache_stamp: Optional[tuple] = None

    def save(
        self,
        *,
        collection_name: str,
        model: CollectionModelConfig,
    ) -> CollectionMetadata:
        normalized_name = self._normalize_collection_name(collection_name)

        with self._exclusive_access():
            collections = self._read_all(use_cache=False)

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
                framing=model.framing,
                views=model.views,
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

    def list(self) -> list:
        collections = self._read_all()
        return [
            self._metadata_from_dict(collections[name]) for name in sorted(collections)
        ]

    def delete(self, collection_name: str) -> CollectionMetadata:
        normalized_name = self._normalize_collection_name(collection_name)

        with self._exclusive_access():
            collections = self._read_all(use_cache=False)

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

    @contextmanager
    def _exclusive_access(self):
        """Serialize read-modify-write cycles across threads and processes."""
        with self._thread_lock:
            lock_path = self.metadata_path.with_name(self.metadata_path.name + ".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)

            with lock_path.open("w") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _file_stamp(self) -> Optional[tuple]:
        try:
            stat_result = self.metadata_path.stat()
        except OSError:
            return None
        return (stat_result.st_mtime_ns, stat_result.st_size)

    def _read_all(self, *, use_cache: bool = True) -> dict:
        stamp = self._file_stamp()

        if use_cache and self._cache is not None and stamp == self._cache_stamp:
            return dict(self._cache)

        if stamp is None:
            return {}

        with self.metadata_path.open("r", encoding="utf-8") as metadata_file:
            data = json.load(metadata_file)

        if not isinstance(data, dict):
            return {}

        self._cache = data
        self._cache_stamp = stamp
        return dict(data)

    def _write_all(self, collections: dict) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.metadata_path.with_suffix(f"{self.metadata_path.suffix}.tmp")

        with temp_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(collections, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")

        temp_path.replace(self.metadata_path)
        self._cache = collections
        self._cache_stamp = self._file_stamp()

    def _metadata_from_dict(self, data: dict) -> CollectionMetadata:
        return CollectionMetadata(
            collection_name=str(data["collection_name"]),
            embedding_provider=str(data["embedding_provider"]),
            embedding_model=str(data["embedding_model"]),
            embedding_pretrained=str(data["embedding_pretrained"]),
            vector_size=int(data["vector_size"]),
            distance=str(data["distance"]),
            created_at=str(data["created_at"]),
            # Collections written before framing existed used the stock
            # centre-crop, so that is what they must keep being searched with.
            framing=str(data.get("framing", "crop")),
            views=int(data.get("views", 1)),
        )

    def _normalize_collection_name(self, collection_name: str) -> str:
        return collection_name.strip()
