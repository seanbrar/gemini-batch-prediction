"""
Gemini Batch Processing Framework
"""

from .batch_processor import BatchProcessor
from .client import GeminiClient
from .exceptions import APIError, GeminiBatchError

__version__ = "0.1.0"
__all__ = [
    "GeminiClient",
    "BatchProcessor",
    # Exceptions
    "GeminiBatchError",
    "APIError",
]
