from functools import lru_cache
from typing import Optional

from app.core.config import Settings, get_settings
from app.embedding.base import EmbeddingModelMetadata, EmbeddingProvider
from app.embedding.registry import EmbeddingProviderRegistry, get_embedding_registry


class EmbeddingManager:
    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        registry: Optional[EmbeddingProviderRegistry] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or get_embedding_registry()
        self._provider_cache: dict[tuple[str, str, str, int], EmbeddingProvider] = {}

    def get_default_provider(self) -> EmbeddingProvider:
        return self.get_provider(
            provider_name=self.settings.default_embedding_provider,
            model_name=self.settings.default_model_name,
            model_pretrained=self.settings.default_model_pretrained,
            vector_size=self.settings.default_vector_size,
        )

    def get_provider(
        self,
        *,
        provider_name: str,
        model_name: str,
        model_pretrained: str,
        vector_size: int,
    ) -> EmbeddingProvider:
        cache_key = (
            provider_name.strip().lower(),
            model_name,
            model_pretrained,
            vector_size,
        )

        if cache_key not in self._provider_cache:
            provider_class = self.registry.get(provider_name)
            self._provider_cache[cache_key] = provider_class(
                model_name=model_name,
                model_pretrained=model_pretrained,
                vector_size=vector_size,
            )

        return self._provider_cache[cache_key]

    def get_default_model_metadata(self) -> EmbeddingModelMetadata:
        return self.get_default_provider().metadata

    def list_available_models(self) -> list[EmbeddingModelMetadata]:
        return [self.get_default_model_metadata()]


@lru_cache
def get_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager()
