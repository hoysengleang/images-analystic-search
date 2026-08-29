from inspect import signature
from typing import Optional

from app.core.config import Settings
from app.embedding.base import EmbeddingModelMetadata, EmbeddingProvider
from app.embedding.registry import EmbeddingProviderRegistry


class EmbeddingManager:
    def __init__(
        self,
        *,
        settings: Settings,
        registry: EmbeddingProviderRegistry,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self._provider_cache: dict[tuple[str, str, str, int], EmbeddingProvider] = {}

    def get_default_provider(self) -> EmbeddingProvider:
        return self.get_provider(
            provider_name=self.settings.default_embedding_provider,
            model_name=self.settings.default_model_name,
            model_pretrained=self.settings.default_model_pretrained,
            vector_size=self.settings.default_vector_size,
            framing=self.settings.image_framing,
            views=self.settings.embed_views,
        )

    def get_provider(
        self,
        *,
        provider_name: str,
        model_name: str,
        model_pretrained: str,
        vector_size: int,
        framing: Optional[str] = None,
        views: Optional[int] = None,
    ) -> EmbeddingProvider:
        """Return the provider a collection was built with.

        Framing and view count change the vector space just as the model does,
        so they are part of the cache key and are passed through to providers
        that understand them.
        """
        cache_key = (
            provider_name.strip().lower(),
            model_name,
            model_pretrained,
            vector_size,
            framing,
            views,
        )

        if cache_key not in self._provider_cache:
            provider_class = self.registry.get(provider_name)
            self._provider_cache[cache_key] = provider_class(
                model_name=model_name,
                model_pretrained=model_pretrained,
                vector_size=vector_size,
                # Providers that do not offer framing keep their own defaults.
                # Checked by signature rather than by catching TypeError, which
                # would also swallow a real error from inside the constructor.
                **self._supported_options(
                    provider_class,
                    {"framing": framing, "views": views},
                ),
            )

        return self._provider_cache[cache_key]

    @staticmethod
    def _supported_options(provider_class, options: dict) -> dict:
        parameters = signature(provider_class.__init__).parameters
        return {
            name: value
            for name, value in options.items()
            if value is not None and name in parameters
        }

    def get_default_model_metadata(self) -> EmbeddingModelMetadata:
        return self.get_default_provider().metadata

    def list_available_models(self) -> list[EmbeddingModelMetadata]:
        return [self.get_default_model_metadata()]
