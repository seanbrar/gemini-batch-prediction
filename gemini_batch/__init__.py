"""
Gemini Batch Processing Framework
"""

import importlib.metadata

from .batch_processor import BatchProcessor
from .client import GeminiClient
from .exceptions import APIError, GeminiBatchError, MissingKeyError, NetworkError

try:
    __version__ = importlib.metadata.version("gemini-batch")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.2.0"  # fallback version
__all__ = [
    "GeminiClient",
    "BatchProcessor",
    # Exceptions
    "GeminiBatchError",
    "APIError",
    "MissingKeyError",
    "NetworkError",
]
