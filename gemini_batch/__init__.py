"""
Gemini Batch Processing Framework
"""

import importlib.metadata

from .batch_processor import BatchProcessor
from .config import ConfigManager
from .exceptions import APIError, GeminiBatchError, MissingKeyError, NetworkError
from .files import FileOperations, FileType
from .gemini_client import GeminiClient

try:
    __version__ = importlib.metadata.version("gemini-batch")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.4.0"  # fallback version
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
