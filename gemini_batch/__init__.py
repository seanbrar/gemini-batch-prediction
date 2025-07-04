"""
Gemini Batch Processing Framework
"""

import importlib.metadata

from .batch_processor import BatchProcessor
from .client.cache_manager import CacheInfo, CacheManager, CacheStrategy
from .config import ConfigManager
from .conversation import (
    ConversationSession,
    ConversationTurn,
    create_conversation,
    load_conversation,
)
from .exceptions import APIError, GeminiBatchError, MissingKeyError, NetworkError
from .files import FileType
from .gemini_client import GeminiClient

try:
    __version__ = importlib.metadata.version("gemini-batch")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.5.0"  # fallback version

__all__ = [
    "GeminiClient",
    "BatchProcessor",
    "ConfigManager",
    # File type enum
    "FileType",
    # Exceptions
    "GeminiBatchError",
    "APIError",
    "MissingKeyError",
    "NetworkError",
    # Advanced cache management
    "CacheManager",
    "CacheStrategy",
    "CacheInfo",
    # Conversation
    "ConversationSession",
    "ConversationTurn",
    "create_conversation",
    "load_conversation",
]
