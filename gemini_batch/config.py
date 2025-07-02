"""
Configuration system for rate limiting and model capabilities
"""

from dataclasses import dataclass
from enum import Enum
import os
from typing import Dict, Optional, Union

from .exceptions import GeminiBatchError, MissingKeyError


class APITier(Enum):
    FREE = "free"
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


class CachingRecommendation(Enum):
    """Recommended caching strategy for a given context"""

    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
    NONE = "none"


@dataclass
class CachingCapabilities:
    """Caching capabilities for a model"""

    supports_implicit: bool = False
    supports_explicit: bool = True  # Default to explicit for all caching-enabled models
    implicit_minimum_tokens: Optional[int] = None
    explicit_minimum_tokens: int = 4096

    def can_use_implicit_caching(self, token_count: int) -> bool:
        """Check if implicit caching is available for given token count"""
        return (
            self.supports_implicit
            and self.implicit_minimum_tokens is not None
            and token_count >= self.implicit_minimum_tokens
        )

    def can_use_explicit_caching(self, token_count: int) -> bool:
        """Check if explicit caching is available for given token count"""
        return self.supports_explicit and token_count >= self.explicit_minimum_tokens


@dataclass
class ModelCapabilities:
    """Intrinsic model capabilities (same across all tiers)"""

    supports_multimodal: bool
    context_window: int
    caching: Optional[CachingCapabilities] = None

    def get_caching(self) -> CachingCapabilities:
        """Get caching capabilities with fallback for backwards compatibility"""
        if self.caching is not None:
            return self.caching

        # No caching support at all if no explicit caching configuration
        return CachingCapabilities(
            supports_implicit=False,
            supports_explicit=False,
            explicit_minimum_tokens=float("inf"),  # Impossible threshold
        )


@dataclass
class RateLimits:
    """Tier-specific rate limits for a model"""

    requests_per_minute: int
    tokens_per_minute: int


@dataclass
class ModelLimits:
    """Combined model capabilities and rate limits"""

    requests_per_minute: int
    tokens_per_minute: int
    supports_multimodal: bool
    context_window: int
    caching: Optional[CachingCapabilities] = None


# Helper factory functions for common caching patterns
def _gemini_25_flash_caching() -> CachingCapabilities:
    """Caching capabilities for Gemini 2.5 Flash models"""
    return CachingCapabilities(
        supports_implicit=True,
        supports_explicit=True,
        implicit_minimum_tokens=1024,
        explicit_minimum_tokens=1024,
    )


def _gemini_25_pro_caching() -> CachingCapabilities:
    """Caching capabilities for Gemini 2.5 Pro models"""
    return CachingCapabilities(
        supports_implicit=True,
        supports_explicit=True,
        implicit_minimum_tokens=2048,
        explicit_minimum_tokens=2048,
    )


def _explicit_only_caching(minimum_tokens: int = 4096) -> CachingCapabilities:
    """Caching capabilities for explicit-only models (2.0, 1.5 family)"""
    return CachingCapabilities(
        supports_implicit=False,
        supports_explicit=True,
        explicit_minimum_tokens=minimum_tokens,
    )


# Base model capabilities (shared across tiers)
MODEL_CAPABILITIES = {
    "gemini-2.5-flash-preview-05-20": ModelCapabilities(
        supports_multimodal=True,
        context_window=1_000_000,
        caching=_gemini_25_flash_caching(),
    ),
    "gemini-2.5-pro-preview-06-05": ModelCapabilities(
        supports_multimodal=True,
        context_window=2_000_000,
        caching=_gemini_25_pro_caching(),
    ),
    "gemini-2.0-flash": ModelCapabilities(
        supports_multimodal=True,
        context_window=1_000_000,
        caching=_explicit_only_caching(),
    ),
    "gemini-2.0-flash-lite": ModelCapabilities(
        supports_multimodal=True,
        context_window=1_000_000,
        caching=_explicit_only_caching(),
    ),
    "gemini-1.5-flash": ModelCapabilities(
        supports_multimodal=True,
        context_window=1_000_000,
        caching=_explicit_only_caching(),
    ),
    "gemini-1.5-flash-8b": ModelCapabilities(
        supports_multimodal=True,
        context_window=1_000_000,
        caching=_explicit_only_caching(),
    ),
    "gemini-1.5-pro": ModelCapabilities(
        supports_multimodal=True,
        context_window=2_000_000,
        caching=_explicit_only_caching(),
    ),
}

# Tier-specific rate limits (only what varies by tier)
TIER_RATE_LIMITS = {
    APITier.FREE: {
        "gemini-2.5-flash-preview-05-20": RateLimits(10, 250_000),
        "gemini-2.0-flash": RateLimits(15, 1_000_000),
        "gemini-2.0-flash-lite": RateLimits(30, 1_000_000),
        "gemini-1.5-flash": RateLimits(15, 250_000),
        "gemini-1.5-flash-8b": RateLimits(15, 250_000),
    },
    APITier.TIER_1: {
        "gemini-2.5-flash-preview-05-20": RateLimits(1_000, 1_000_000),
        "gemini-2.5-pro-preview-06-05": RateLimits(150, 2_000_000),
        "gemini-2.0-flash": RateLimits(2_000, 4_000_000),
        "gemini-2.0-flash-lite": RateLimits(4_000, 4_000_000),
        "gemini-1.5-flash": RateLimits(2_000, 4_000_000),
        "gemini-1.5-flash-8b": RateLimits(4_000, 4_000_000),
        "gemini-1.5-pro": RateLimits(1_000, 4_000_000),
    },
    APITier.TIER_2: {
        "gemini-2.5-flash-preview-05-20": RateLimits(2_000, 3_000_000),
        "gemini-2.5-pro-preview-06-05": RateLimits(1_000, 5_000_000),
        "gemini-2.0-flash": RateLimits(10_000, 10_000_000),
        "gemini-2.0-flash-lite": RateLimits(20_000, 10_000_000),
        "gemini-1.5-flash": RateLimits(2_000, 4_000_000),
        "gemini-1.5-flash-8b": RateLimits(4_000, 4_000_000),
        "gemini-1.5-pro": RateLimits(1_000, 4_000_000),
    },
    APITier.TIER_3: {
        "gemini-2.5-flash-preview-05-20": RateLimits(10_000, 8_000_000),
        "gemini-2.5-pro-preview-06-05": RateLimits(2_000, 8_000_000),
        "gemini-2.0-flash": RateLimits(30_000, 30_000_000),
        "gemini-2.0-flash-lite": RateLimits(30_000, 30_000_000),
    },
}

# Tier display names
TIER_NAMES = {
    APITier.FREE: "Free Tier",
    APITier.TIER_1: "Tier 1 (Billing Enabled)",
    APITier.TIER_2: "Tier 2",
    APITier.TIER_3: "Tier 3",
}


def _validate_api_key(api_key: str) -> None:
    """Validate API key format and raise appropriate errors"""
    if not api_key or not isinstance(api_key, str):
        raise MissingKeyError("API key must be a non-empty string")

    api_key = api_key.strip()
    if len(api_key) < 30:
        raise MissingKeyError("API key appears to be invalid (too short)")


def _parse_tier_from_string(tier_str: str) -> Optional[APITier]:
    """Parse tier string to APITier enum with validation"""
    if not tier_str:
        return None

    tier_str = tier_str.lower().strip()

    # Handle common variations
    tier_mapping = {
        "free": APITier.FREE,
        "tier1": APITier.TIER_1,
        "tier_1": APITier.TIER_1,
        "tier-1": APITier.TIER_1,
        "tier2": APITier.TIER_2,
        "tier_2": APITier.TIER_2,
        "tier-2": APITier.TIER_2,
        "tier3": APITier.TIER_3,
        "tier_3": APITier.TIER_3,
        "tier-3": APITier.TIER_3,
    }

    return tier_mapping.get(tier_str)


class ConfigManager:
    """Manages API configuration including environment variable integration"""

    def __init__(
        self,
        tier: Optional[APITier] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize configuration with explicit parameters or environment fallback"""
        # API key resolution and validation
        self.api_key = api_key or self._get_api_key_from_env()
        if self.api_key:
            _validate_api_key(self.api_key)

        # Tier resolution with environment fallback
        self.tier = tier or self._parse_tier_from_env() or self._detect_tier()

        # Model resolution with environment fallback
        self.model = model or self._get_model_from_env() or self._get_default_model()

        # Validate that the selected model is available in the selected tier
        if self.model and not self.get_model_limits(self.model):
            available_models = self._get_available_models()
            raise GeminiBatchError(
                f"Model '{self.model}' is not available in {self.tier.value} tier. "
                f"Available models: {', '.join(available_models)}"
            )

    @classmethod
    def from_env(cls) -> "ConfigManager":
        """Factory for pure environment-driven configuration"""
        return cls()

    @classmethod
    def for_testing(
        cls, tier: APITier = APITier.FREE, model: str = "gemini-2.0-flash"
    ) -> "ConfigManager":
        """Factory for testing with sensible defaults (no API key required)"""
        return cls(tier=tier, model=model, api_key="test-key-" + "x" * 30)

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables"""
        return os.getenv("GEMINI_API_KEY")

    def _parse_tier_from_env(self) -> Optional[APITier]:
        """Parse tier from environment variable with validation"""
        tier_str = os.getenv("GEMINI_TIER")
        if not tier_str:
            return None

        parsed_tier = _parse_tier_from_string(tier_str)
        if parsed_tier is None:
            # Don't fail hard - just warn and fall back
            print(f"Warning: Invalid GEMINI_TIER '{tier_str}'. Using tier detection.")

        return parsed_tier

    def _get_model_from_env(self) -> Optional[str]:
        """Get model name from environment variable"""
        return os.getenv("GEMINI_MODEL")

    def _detect_tier(self) -> APITier:
        """
        Attempt to detect API tier through feature testing
        Falls back to FREE tier if detection fails
        """
        # For now, default to FREE - could be enhanced with actual detection
        # via test API calls or environment configuration
        return APITier.FREE

    def _get_default_model(self) -> str:
        """Get default model for detected tier - optimized O(1) access"""
        tier_limits = TIER_RATE_LIMITS.get(self.tier, {})
        return next(iter(tier_limits), "gemini-2.0-flash")

    def _get_available_models(self) -> list[str]:
        """Get list of available models for current tier - cached for error messages"""
        if not hasattr(self, "_cached_available_models"):
            self._cached_available_models = list(
                TIER_RATE_LIMITS.get(self.tier, {}).keys()
            )
        return self._cached_available_models

    def get_model_limits(self, model: str) -> Optional[ModelLimits]:
        """Get combined rate limits and capabilities for specific model and tier"""
        # Get tier-specific rate limits
        tier_limits = TIER_RATE_LIMITS.get(self.tier, {})
        rate_limits = tier_limits.get(model)
        if not rate_limits:
            return None

        # Get model capabilities
        capabilities = MODEL_CAPABILITIES.get(model)
        if not capabilities:
            return None

        # Combine into ModelLimits
        return ModelLimits(
            requests_per_minute=rate_limits.requests_per_minute,
            tokens_per_minute=rate_limits.tokens_per_minute,
            supports_multimodal=capabilities.supports_multimodal,
            context_window=capabilities.context_window,
            caching=capabilities.caching,
        )

    def select_optimal_model(
        self, content_tokens: int, query_count: int, video_count: int = 1
    ) -> str:
        """
        Enhanced model selection considering tier capabilities
        TODO: Implement intelligent model selection logic based on requirements
        """
        # For now, just return the default model for the tier
        return self.model

    def get_rate_limiter_config(self, model: str) -> Dict:
        """Get rate limiter configuration for model"""
        limits = self.get_model_limits(model)
        if not limits:
            # If model not found in current tier, raise an error
            available_models = self._get_available_models()
            raise GeminiBatchError(
                f"Model '{model}' is not available in {self.tier.value} tier. "
                f"Available models: {', '.join(available_models)}"
            )

        return {
            "requests_per_minute": limits.requests_per_minute,
            "tokens_per_minute": limits.tokens_per_minute,
        }

    def get_tier_name(self) -> str:
        """Get human-readable tier name"""
        return TIER_NAMES.get(self.tier, "Unknown Tier")

    def requires_api_key(self) -> bool:
        """Check if current configuration requires an API key"""
        return self.api_key is not None

    def can_use_caching(
        self, model: str, token_count: int, prefer_implicit: bool = True
    ) -> Dict[str, Union[bool, str]]:
        """Check caching availability for a model and token count"""
        capabilities = MODEL_CAPABILITIES.get(model)
        if not capabilities:
            return {
                "implicit": False,
                "explicit": False,
                "recommended": CachingRecommendation.NONE.value,
            }

        caching = capabilities.get_caching()

        implicit_available = caching.can_use_implicit_caching(token_count)
        explicit_available = caching.can_use_explicit_caching(token_count)

        # Determine recommendation using enum
        if implicit_available and prefer_implicit:
            recommendation = CachingRecommendation.IMPLICIT
        elif explicit_available:
            recommendation = CachingRecommendation.EXPLICIT
        else:
            recommendation = CachingRecommendation.NONE

        return {
            "implicit": implicit_available,
            "explicit": explicit_available,
            "recommended": recommendation.value,
        }

    def get_caching_thresholds(self, model: str) -> Dict[str, Optional[int]]:
        """Get minimum token thresholds for caching types"""
        capabilities = MODEL_CAPABILITIES.get(model)
        if not capabilities:
            return {"implicit": None, "explicit": None}

        caching = capabilities.get_caching()
        return {
            "implicit": caching.implicit_minimum_tokens
            if caching.supports_implicit
            else None,
            "explicit": caching.explicit_minimum_tokens
            if caching.supports_explicit
            else None,
        }

    def get_config_summary(self) -> Dict[str, str]:
        """Get summary of current configuration for debugging"""
        return {
            "tier": self.tier.value,
            "tier_name": self.get_tier_name(),
            "model": self.model,
            "api_key_present": bool(self.api_key),
            "api_key_length": len(self.api_key) if self.api_key else 0,
        }
