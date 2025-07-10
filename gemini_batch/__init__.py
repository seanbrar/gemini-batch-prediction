"""
Gemini Batch Processing Framework
"""

import importlib.metadata
import logging

from .analysis.schema_analyzer import SchemaAnalyzer
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
from .prompts import BatchPromptBuilder, StructuredPromptBuilder

# Set up a null handler for the library's root logger.
# This prevents 'No handler found' errors if the consuming app has no logging configured.
logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = importlib.metadata.version("gemini-batch")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.6.1"  # fallback version

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
    # Structured output components
    "BatchPromptBuilder",
    "StructuredPromptBuilder",
    "SchemaAnalyzer",
]
