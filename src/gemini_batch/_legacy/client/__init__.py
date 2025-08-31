"""Supporting components for Gemini API client

Note: The main GeminiClient is now at gemini_batch.GeminiClient (root level)
for better hierarchical architecture. This package contains supporting components.
"""  # noqa: D415

from .cache_manager import (
    CacheInfo,
    CacheManager,
    CacheMetrics,
    CacheResult,
    CacheStrategy,
)
from .configuration import RateLimitConfig

# Specialized components for custom integrations
from .content_processor import ContentProcessor
from .error_handler import GenerationErrorHandler
from .file_upload_manager import FileUploadManager
from .prompt_builder import PromptBuilder
from .rate_limiter import RateLimiter
from .token_counter import TokenCounter

# Public API - supporting components only
__all__ = [  # noqa: RUF022
    # Configuration
    "RateLimitConfig",
    # Advanced components (for custom extensions)
    "ContentProcessor",
    "FileUploadManager",
    "GenerationErrorHandler",
    "PromptBuilder",
    "RateLimiter",
    "TokenCounter",
    # Cache management
    "CacheManager",
    "CacheStrategy",
    "CacheResult",
    "CacheInfo",
    "CacheMetrics",
]
