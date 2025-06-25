"""
Supporting components for Gemini API client

Note: The main GeminiClient is now at gemini_batch.GeminiClient (root level)
for better hierarchical architecture. This package contains supporting components.

For backward compatibility, use: from gemini_batch import GeminiClient
"""

# Configuration classes for advanced usage
from .configuration import ClientConfiguration, RateLimitConfig

# Specialized components for custom integrations
from .content_processor import ContentProcessor
from .error_handler import GenerationErrorHandler
from .prompt_builder import PromptBuilder
from .rate_limiter import RateLimiter

# Public API - supporting components only
# For GeminiClient, use: from gemini_batch import GeminiClient
__all__ = [
    # Configuration
    "ClientConfiguration",
    "RateLimitConfig",
    # Advanced components (for custom extensions)
    "ContentProcessor",
    "GenerationErrorHandler",
    "PromptBuilder",
    "RateLimiter",
]
