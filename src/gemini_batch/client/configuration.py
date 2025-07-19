"""Client configuration handling for Gemini API integration"""

from dataclasses import dataclass

from ..constants import RATE_LIMIT_WINDOW


@dataclass
class RateLimitConfig:
    """Rate limiting parameters for API request throttling"""

    requests_per_minute: int
    tokens_per_minute: int
    window_seconds: int = RATE_LIMIT_WINDOW
