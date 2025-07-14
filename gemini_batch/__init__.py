"""
Gemini Batch Processing Framework
"""

import importlib.metadata
import logging
from typing import Any, List, Unpack

from .analysis.schema_analyzer import SchemaAnalyzer
from .batch_processor import BatchProcessor
from .config import (
    ConversationConfig,
    GeminiConfig,
    config_scope,
    debug_config,
    get_config,
    get_effective_config,
)
from .conversation import (
    ConversationSession,
    ConversationTurn,  # Keep for type hinting
    create_conversation,
    load_conversation,
)
from .exceptions import APIError, GeminiBatchError, MissingKeyError, NetworkError
from .files import FileType
from .gemini_client import GeminiClient
from .telemetry import TelemetryContext

# Version handling
try:
    __version__ = importlib.metadata.version("gemini-batch")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.7.0"  # fallback version

# Set up a null handler for the library's root logger.
# This prevents 'No handler found' errors if the consuming app has no logging configured.
logging.getLogger(__name__).addHandler(logging.NullHandler())


def process_questions(
    content: Any, questions: List[str], **config: Unpack[GeminiConfig]
) -> dict:
    """
    Process questions against content in a single call. Perfect for scripts and notebooks.
    """
    processor = BatchProcessor(**config)
    return processor.process_questions(content, questions)


# Public API
__all__ = [
    # Core Classes
    "GeminiClient",
    "BatchProcessor",
    "ConversationSession",
    # Factory Functions
    "create_conversation",
    "load_conversation",
    # One-shot function
    "process_questions",
    # Configuration
    "config_scope",
    "get_config",
    "get_effective_config",
    "debug_config",
    "GeminiConfig",
    "ConversationConfig",
    # Supporting Types & Exceptions
    "ConversationTurn",
    "FileType",
    "GeminiBatchError",
    "APIError",
    "MissingKeyError",
    "NetworkError",
    "SchemaAnalyzer",
    "TelemetryContext",
]
