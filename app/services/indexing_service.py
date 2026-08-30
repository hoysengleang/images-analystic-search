from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from app.core.config import Settings
from app.core.errors import BadRequestError, OpenVisionSearchError
from app.embedding.manager import EmbeddingManager, get_configured_provider
from app.schemas.index import IndexImageItem, IndexRequest, IndexResponse
from app.services.collection_metadata_service import (
    CollectionMetadata,
    CollectionMetadataService,
)
from app.services.image_loader import ImageLoader
from app.services.vector_service import VectorRecord, VectorService
from app.utils.image_utils import normalize_image_extension, validate_and_load_image
from app.utils.path_utils import resolve_safe_image_path

if TYPE_CHECKING:
    from PIL import Image

T = TypeVar("T")


@dataclass(frozen=True)
class ImageSourceRef:
    """How an indexed image was obtained, stored on the vector payload."""

    type: str
    value: str


@dataclass(frozen=True)
class PendingImage:
    """A decoded image waiting to be embedded and stored."""

    image_id: str
    source: ImageSourceRef
    metadata: dict
    image: Image.Image


class IndexingService:
    """Loads images from any source, embeds them, and stores the vectors.

    Loading is per image so one bad file cannot fail the request, while
    embedding and storing run in batches because that is where the time goes.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        metadata_service: CollectionMetadataService,
        image_loader: ImageLoader,
        embedding_manager: EmbeddingManager,
        vector_service: VectorService,
    ) -> None:
        self.settings = settings
        self.metadata_service = metadata_service
        self.image_loader = image_loader
        self.embedding_manager = embedding_manager
        self.vector_service = vector_service

    def index(self, request: IndexRequest) -> IndexResponse:
        collection_metadata = self.metadata_service.get(request.collection_name)
        errors: list[dict[str, Any]] = []

        return self._embed_and_store(
            collection_metadata,
            self._load_requested_images(request.images, errors),
            errors,
        )

    def index_upload(
        self,
        *,
        collection_name: str,
        image_id: str,
        image_bytes: bytes,
        filename: str,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> IndexResponse:
        collection_metadata = self.metadata_service.get(collection_name)

        validated_image = validate_and_load_image(
            image_bytes,
            max_size_mb=self.settings.max_image_size_mb,
            extension=filename,
            allowed_extensions=self.settings.allowed_image_extensions,
            max_pixels=self.settings.max_image_pixels,
        )

        upload_metadata = dict(metadata or {})
        if filename:
            upload_metadata.setdefault("original_filename", filename)
        if content_type:
            upload_metadata.setdefault("content_type", content_type)

        pending = (
            PendingImage(
                image_id=image_id,
                source=ImageSourceRef(type="upload", value=filename or image_id),
                metadata=upload_metadata,
                image=validated_image.image,
            ),
        )
        return self._embed_and_store(collection_metadata, pending, [])

    def index_folder(
        self,
        *,
        collection_name: str,
        folder_path: str,
        recursive: bool,
        metadata: Optional[dict[str, Any]] = None,
    ) -> IndexResponse:
        collection_metadata = self.metadata_service.get(collection_name)
        resolved_folder = resolve_safe_image_path(
            folder_path,
            self.settings.allowed_image_root,
        )

        if not resolved_folder.is_dir():
            raise BadRequestError(
                message="Folder path must point to a directory",
                code="INVALID_IMAGE_FOLDER",
                details={"folder_path": folder_path},
            )

        errors: list[dict[str, Any]] = []

        return self._embed_and_store(
            collection_metadata,
            self._load_folder_images(
                resolved_folder,
                recursive=recursive,
                metadata=metadata,
                errors=errors,
            ),
            errors,
        )

    def _load_requested_images(
        self,
        items: Iterable[IndexImageItem],
        errors: list[dict[str, Any]],
    ) -> Iterator[PendingImage]:
        for item in items:
            try:
                image = self.image_loader.load_from_source(item.source)
            except Exception as exc:
                errors.append(self._error_for(item.id, exc))
                continue

            yield PendingImage(
                image_id=item.id,
                source=ImageSourceRef(type=item.source.type, value=item.source.value),
                metadata=self._metadata_for_item(item),
                image=image,
            )

    def _load_folder_images(
        self,
        folder_path: Path,
        *,
        recursive: bool,
        metadata: Optional[dict[str, Any]],
        errors: list[dict[str, Any]],
    ) -> Iterator[PendingImage]:
        for image_path in self._iter_image_paths(folder_path, recursive=recursive):
            image_id = self._image_id_from_folder_path(folder_path, image_path)
            try:
                image = self.image_loader.load_from_path(str(image_path))
            except Exception as exc:
                errors.append(self._error_for(image_id, exc))
                continue

            item_metadata = dict(metadata or {})
            item_metadata.setdefault(
                "relative_path",
                str(image_path.relative_to(folder_path)),
            )
            item_metadata.setdefault("filename", image_path.name)

            yield PendingImage(
                image_id=image_id,
                source=ImageSourceRef(type="folder", value=str(image_path)),
                metadata=item_metadata,
                image=image,
            )

    def _embed_and_store(
        self,
        collection_metadata: CollectionMetadata,
        pending_images: Iterable[PendingImage],
        errors: list[dict[str, Any]],
    ) -> IndexResponse:
        """Embed and store images batch by batch.

        ``pending_images`` is consumed lazily and each batch is released before
        the next is loaded, so peak memory is bounded by ``EMBED_BATCH_SIZE``
        rather than by the size of the folder or request being indexed.
        """
        embedding_provider = get_configured_provider(
            self.embedding_manager,
            collection_metadata,
        )
        indexed_ids: list[str] = []

        for batch in self._batched(pending_images, self.settings.embed_batch_size):
            try:
                vectors = embedding_provider.embed_images(
                    [pending.image for pending in batch]
                )
                self.vector_service.upsert_images(
                    collection_name=collection_metadata.collection_name,
                    records=[
                        VectorRecord(
                            image_id=pending.image_id,
                            vector=vector,
                            source=pending.source,
                            metadata=pending.metadata,
                        )
                        for pending, vector in zip(batch, vectors)
                    ],
                    embedding=collection_metadata.to_embedding_metadata(),
                )
            except Exception as exc:
                errors.extend(self._error_for(pending.image_id, exc) for pending in batch)
                continue
            finally:
                # Decoded images can be megabytes each; do not wait for the GC.
                for pending in batch:
                    pending.image.close()

            indexed_ids.extend(pending.image_id for pending in batch)

        return IndexResponse(
            collection_name=collection_metadata.collection_name,
            indexed_count=len(indexed_ids),
            failed_count=len(errors),
            ids=indexed_ids,
            errors=errors,
        )

    def _error_for(self, image_id: str, exc: Exception) -> dict[str, Any]:
        if isinstance(exc, OpenVisionSearchError):
            return {
                "id": image_id,
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }

        return {
            "id": image_id,
            "code": "INDEX_IMAGE_FAILED",
            "message": "Could not index image",
            "details": {"error": str(exc)},
        }

    def _metadata_for_item(self, item: IndexImageItem) -> dict[str, Any]:
        metadata = dict(item.metadata)
        display_image_url = item.display_image_url or item.source.display_image_url
        if display_image_url:
            metadata.setdefault("display_image_url", display_image_url)
        return metadata

    def _batched(self, items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
        """Chunk a lazy iterable, holding at most one batch at a time."""
        batch: list[T] = []
        for item in items:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def _iter_image_paths(
        self,
        folder_path: Path,
        *,
        recursive: bool,
    ) -> list[Path]:
        allowed_extensions = self.settings.allowed_image_extensions
        paths = folder_path.rglob("*") if recursive else folder_path.glob("*")
        return sorted(
            path
            for path in paths
            if path.is_file()
            and normalize_image_extension(path.suffix) in allowed_extensions
        )

    def _image_id_from_folder_path(self, folder_path: Path, image_path: Path) -> str:
        relative_path = image_path.relative_to(folder_path)
        return str(relative_path.with_suffix("")).replace("/", "__")
