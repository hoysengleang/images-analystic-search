"""Embedding provider implementations package."""

from app.embedding.providers.onnx_provider import ONNXProvider
from app.embedding.providers.openclip_provider import OpenCLIPProvider

__all__ = ["ONNXProvider", "OpenCLIPProvider"]
