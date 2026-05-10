from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.errors import OpenVisionSearchError
from app.embedding.base import EmbeddingModelMetadata
from app.embedding.manager import EmbeddingManager, get_embedding_manager
from app.schemas.index import IndexImageItem, IndexRequest, IndexResponse
from app.services.collection_metadata_service import (
    CollectionMetadata,
    CollectionMetadataService,
    get_collection_metadata_service,
)
from app.services.image_loader import ImageLoader, get_image_loader
from app.services.vector_service import VectorService, get_vector_service
from app.utils.image_utils import (
    normalize_image_extension,
    validate_and_load_image,
)
from app.utils.path_utils import resolve_safe_image_path


@dataclass(frozen=True)
class UploadImageSource:
    type: str
    value: str


@dataclass(frozen=True)
class FolderImageSource:
    type: str
    value: str


class IndexingService:
    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        metadata_service: Optional[CollectionMetadataService] = None,
        image_loader: Optional[ImageLoader] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        vector_service: Optional[VectorService] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.metadata_service = metadata_service or get_collection_metadata_service()
        self.image_loader = image_loader or get_image_loader()
        self.embedding_manager = embedding_manager or get_embedding_manager()
        self.vector_service = vector_service or get_vector_service()

    def index(self, request: IndexRequest) -> IndexResponse:
        collection_metadata = self.metadata_service.get(request.collection_name)
        embedding_provider = self.embedding_manager.get_provider(
            provider_name=collection_metadata.embedding_provider,
            model_name=collection_metadata.embedding_model,
            model_pretrained=collection_metadata.embedding_pretrained,
            vector_size=collection_metadata.vector_size,
        )
        embedding_metadata = self._embedding_metadata_from_collection(
            collection_metadata,
        )

        indexed_ids: list[str] = []
        errors: list[dict[str, object]] = []

        for item in request.images:
            try:
                image = self.image_loader.load_from_source(item.source)
                vector = embedding_provider.embed_image(image)
                self.vector_service.upsert_image(
                    collection_name=collection_metadata.collection_name,
                    image_id=item.id,
                    vector=vector,
                    source=item.source,
                    metadata=self._metadata_for_item(item),
                    embedding=embedding_metadata,
                )
            except OpenVisionSearchError as exc:
                errors.append(
                    {
                        "id": item.id,
                        "code": exc.code,
                        "message": exc.message,
                        "details": exc.details,
                    }
                )
                continue
            except Exception as exc:
                errors.append(
                    {
                        "id": item.id,
                        "code": "INDEX_IMAGE_FAILED",
                        "message": "Could not index image",
                        "details": {"error": str(exc)},
                    }
                )
                continue

            indexed_ids.append(item.id)

        return IndexResponse(
            collection_name=collection_metadata.collection_name,
            indexed_count=len(indexed_ids),
            failed_count=len(errors),
            ids=indexed_ids,
            errors=errors,
        )

    def index_upload(
        self,
        *,
        collection_name: str,
        image_id: str,
        image_bytes: bytes,
        filename: str,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> IndexResponse:
        collection_metadata = self.metadata_service.get(collection_name)
        embedding_provider = self.embedding_manager.get_provider(
            provider_name=collection_metadata.embedding_provider,
            model_name=collection_metadata.embedding_model,
            model_pretrained=collection_metadata.embedding_pretrained,
            vector_size=collection_metadata.vector_size,
        )
        embedding_metadata = self._embedding_metadata_from_collection(
            collection_metadata,
        )

        validated_image = validate_and_load_image(
            image_bytes,
            max_size_mb=self.settings.max_image_size_mb,
            extension=filename,
        )
        vector = embedding_provider.embed_image(validated_image.image)
        upload_metadata = dict(metadata or {})
        if filename:
            upload_metadata.setdefault("original_filename", filename)
        if content_type:
            upload_metadata.setdefault("content_type", content_type)

        self.vector_service.upsert_image(
            collection_name=collection_metadata.collection_name,
            image_id=image_id,
            vector=vector,
            source=UploadImageSource(type="upload", value=filename or image_id),
            metadata=upload_metadata,
            embedding=embedding_metadata,
        )

        return IndexResponse(
            collection_name=collection_metadata.collection_name,
            indexed_count=1,
            failed_count=0,
            ids=[image_id],
            errors=[],
        )

    def index_folder(
        self,
        *,
        collection_name: str,
        folder_path: str,
        recursive: bool,
        metadata: Optional[dict[str, object]] = None,
    ) -> IndexResponse:
        collection_metadata = self.metadata_service.get(collection_name)
        embedding_provider = self.embedding_manager.get_provider(
            provider_name=collection_metadata.embedding_provider,
            model_name=collection_metadata.embedding_model,
            model_pretrained=collection_metadata.embedding_pretrained,
            vector_size=collection_metadata.vector_size,
        )
        embedding_metadata = self._embedding_metadata_from_collection(
            collection_metadata,
        )
        resolved_folder = resolve_safe_image_path(
            folder_path,
            self.settings.allowed_image_root,
        )

        if not resolved_folder.is_dir():
            raise OpenVisionSearchError(
                message="Folder path must point to a directory",
                code="INVALID_IMAGE_FOLDER",
                status_code=400,
                details={"folder_path": folder_path},
            )

        image_paths = self._iter_image_paths(resolved_folder, recursive=recursive)
        indexed_ids: list[str] = []
        errors: list[dict[str, object]] = []

        for image_path in image_paths:
            image_id = self._image_id_from_folder_path(resolved_folder, image_path)
            try:
                image_bytes = image_path.read_bytes()
                validated_image = validate_and_load_image(
                    image_bytes,
                    max_size_mb=self.settings.max_image_size_mb,
                    extension=image_path.suffix,
                )
                vector = embedding_provider.embed_image(validated_image.image)
                item_metadata = dict(metadata or {})
                relative_path = str(image_path.relative_to(resolved_folder))
                item_metadata.setdefault("relative_path", relative_path)
                item_metadata.setdefault("filename", image_path.name)

                self.vector_service.upsert_image(
                    collection_name=collection_metadata.collection_name,
                    image_id=image_id,
                    vector=vector,
                    source=FolderImageSource(type="folder", value=str(image_path)),
                    metadata=item_metadata,
                    embedding=embedding_metadata,
                )
            except OpenVisionSearchError as exc:
                errors.append(
                    {
                        "id": image_id,
                        "code": exc.code,
                        "message": exc.message,
                        "details": exc.details,
                    }
                )
                continue
            except Exception as exc:
                errors.append(
                    {
                        "id": image_id,
                        "code": "INDEX_IMAGE_FAILED",
                        "message": "Could not index image",
                        "details": {"error": str(exc)},
                    }
                )
                continue

            indexed_ids.append(image_id)

        return IndexResponse(
            collection_name=collection_metadata.collection_name,
            indexed_count=len(indexed_ids),
            failed_count=len(errors),
            ids=indexed_ids,
            errors=errors,
        )

    def _metadata_for_item(self, item: IndexImageItem) -> dict[str, object]:
        metadata = dict(item.metadata)
        if item.display_image_url:
            metadata.setdefault("display_image_url", item.display_image_url)
        elif item.source.display_image_url:
            metadata.setdefault("display_image_url", item.source.display_image_url)
        return metadata

    def _embedding_metadata_from_collection(
        self,
        collection_metadata: CollectionMetadata,
    ) -> EmbeddingModelMetadata:
        return EmbeddingModelMetadata(
            provider=collection_metadata.embedding_provider,
            model_name=collection_metadata.embedding_model,
            model_pretrained=collection_metadata.embedding_pretrained,
            vector_size=collection_metadata.vector_size,
        )

    def _iter_image_paths(self, folder_path: Path, *, recursive: bool) -> list[Path]:
        paths = folder_path.rglob("*") if recursive else folder_path.glob("*")
        return sorted(
            path
            for path in paths
            if path.is_file()
            and normalize_image_extension(path.suffix) in {"jpeg", "png", "webp"}
        )

    def _image_id_from_folder_path(self, folder_path: Path, image_path: Path) -> str:
        relative_path = image_path.relative_to(folder_path)
        return str(relative_path.with_suffix("")).replace("/", "__")


def get_indexing_service() -> IndexingService:
    return IndexingService()
