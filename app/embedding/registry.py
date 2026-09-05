from app.core.errors import BadRequestError
from app.embedding.base import EmbeddingProvider
from app.embedding.providers.onnx_provider import ONNXProvider
from app.embedding.providers.openclip_provider import OpenCLIPProvider


class EmbeddingProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, type[EmbeddingProvider]] = {}

    def register(self, provider_class: type[EmbeddingProvider]) -> None:
        provider_name = provider_class.provider_name.strip().lower()
        self._providers[provider_name] = provider_class

    def get(self, provider_name: str) -> type[EmbeddingProvider]:
        normalized_name = provider_name.strip().lower()
        try:
            return self._providers[normalized_name]
        except KeyError as exc:
            raise BadRequestError(
                message=f"Unknown embedding provider: {provider_name}",
                code="UNKNOWN_EMBEDDING_PROVIDER",
                details={"provider": provider_name},
            ) from exc

    def list_provider_names(self) -> list[str]:
        return sorted(self._providers)


embedding_registry = EmbeddingProviderRegistry()
embedding_registry.register(OpenCLIPProvider)
embedding_registry.register(ONNXProvider)


def get_embedding_registry() -> EmbeddingProviderRegistry:
    return embedding_registry
