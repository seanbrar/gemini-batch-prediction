"""
Core utilities for the Gemini Batch Framework
"""

import os
from typing import Optional

from .exceptions import MissingKeyError


def parse_env_bool(key: str, default: bool = False) -> bool:
    """Generic environment boolean parser"""
    value = os.getenv(key, "").lower().strip()
    return value in ("true", "1", "yes", "on") if value else default


def get_env_with_fallback(primary_key: str, fallback_key: str) -> Optional[str]:
    """Generic environment variable with fallback"""
    return os.getenv(primary_key) or os.getenv(fallback_key)


def validate_api_key_format(api_key: str) -> bool:
    """Generic API key format validation - could be used anywhere"""
    if not api_key or not isinstance(api_key, str):
        raise MissingKeyError("API key must be a non-empty string")

    # Basic length check
    api_key = api_key.strip()
    if len(api_key) < 30:  # Too short
        raise MissingKeyError("API key appears to be invalid (too short)")

    return True

