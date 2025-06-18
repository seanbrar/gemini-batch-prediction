"""
Gemini Batch Processing Framework
"""

import importlib.metadata

from .batch_processor import BatchProcessor
from .client import GeminiClient
from .config import ConfigManager
from .exceptions import APIError, GeminiBatchError, MissingKeyError, NetworkError
from .files import FileOperations, FileType

try:
    __version__ = importlib.metadata.version("gemini-batch")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.3.0"  # fallback version
__all__ = [
    "GeminiClient",
    "BatchProcessor",
    "ConfigManager",
    # Files processing
    "FileOperations",
    "FileType",
    # Exceptions
    "GeminiBatchError",
    "APIError",
    "MissingKeyError",
    "NetworkError",
]
