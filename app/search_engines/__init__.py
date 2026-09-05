"""Interchangeable vector search backends."""

from app.search_engines.base import (
    EngineHealth,
    SearchEngine,
    VectorHit,
    VectorSearchRequest,
)
from app.search_engines.native import NativeSearchEngine

__all__ = [
    "EngineHealth",
    "NativeSearchEngine",
    "SearchEngine",
    "VectorHit",
    "VectorSearchRequest",
]
