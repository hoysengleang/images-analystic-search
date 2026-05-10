"""Embedding abstraction package."""

from app.embedding.base import EmbeddingModelMetadata, EmbeddingProvider
from app.embedding.manager import EmbeddingManager

__all__ = ["EmbeddingManager", "EmbeddingModelMetadata", "EmbeddingProvider"]
