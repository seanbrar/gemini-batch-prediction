"""Configuration system for rate limiting and model capabilities"""

from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
from enum import Enum
import logging
import os

# Use typing_extensions for Unpack for Python < 3.11
from typing import Any, Protocol, TypedDict, Unpack

from gemini_batch.constants import MIN_CACHING_THRESHOLD

from .exceptions import GeminiBatchError

log = logging.getLogger(__name__)


# Protocols & Type-Safe Configuration


class APITier(Enum):
    FREE = "free"
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


class GeminiConfig(TypedDict, total=False):
    """Type-safe configuration options for the Gemini client and processor.
    All parameters are optional and will fall back to environment variables or defaults.
    """

    api_key: str
    model: str
    tier: str | APITier
    enable_caching: bool


class ConversationConfig(TypedDict, total=False):
    """Additional, type-safe options specific to conversation sessions."""

    max_history_turns: int


class ClientProtocol(Protocol):
    """Protocol defining the essential capabilities of a content generation client."""

    def generate_content(self, content: Any, prompt: str, **kwargs) -> Any: ...
    def generate_batch(self, content: Any, questions: list[str], **kwargs) -> Any: ...


class ProcessorProtocol(Protocol):
    """Protocol defining the essential capabilities of a question processor."""

    def process_questions(
        self,
        content: Any,
        questions: list[str],
        **kwargs,
    ) -> dict: ...


# Data models for capabilities and limits


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
    implicit_minimum_tokens: int | None = None
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
    caching: CachingCapabilities | None = None

    def get_caching(self) -> CachingCapabilities:
        """Get caching capabilities with fallback"""
        if self.caching is not None:
            return self.caching
        return CachingCapabilities(
            supports_implicit=False,
            supports_explicit=False,
            explicit_minimum_tokens=float("inf"),
        )


@dataclass
class TierRateLimits:
    """Rate limits for a specific model in a specific tier"""

    requests_per_minute: int
    tokens_per_minute: int


@dataclass
class ModelTierLimits:
    """Combined model capabilities and rate limits for a specific tier"""

    requests_per_minute: int
    tokens_per_minute: int
    supports_multimodal: bool
    context_window: int
    caching: CachingCapabilities | None = None


def _explicit_only_caching() -> CachingCapabilities:
    """Standard explicit-only caching configuration"""
    return CachingCapabilities(
        supports_implicit=False,
        supports_explicit=True,
        explicit_minimum_tokens=4096,
    )


def _gemini_25_flash_caching() -> CachingCapabilities:
    """Gemini 2.5 Flash caching configuration with implicit support"""
    return CachingCapabilities(
        supports_implicit=True,
        supports_explicit=True,
        implicit_minimum_tokens=2048,
        explicit_minimum_tokens=4096,
    )


def _gemini_25_pro_caching() -> CachingCapabilities:
    """Gemini 2.5 Pro caching configuration with implicit support"""
    return CachingCapabilities(
        supports_implicit=True,
        supports_explicit=True,
        implicit_minimum_tokens=2048,
        explicit_minimum_tokens=4096,
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
        "gemini-2.5-flash-preview-05-20": TierRateLimits(10, 250_000),
        "gemini-2.0-flash": TierRateLimits(15, 1_000_000),
        "gemini-2.0-flash-lite": TierRateLimits(30, 1_000_000),
        "gemini-1.5-flash": TierRateLimits(15, 250_000),
        "gemini-1.5-flash-8b": TierRateLimits(15, 250_000),
    },
    APITier.TIER_1: {
        "gemini-2.5-flash-preview-05-20": TierRateLimits(1_000, 1_000_000),
        "gemini-2.5-pro-preview-06-05": TierRateLimits(150, 2_000_000),
        "gemini-2.0-flash": TierRateLimits(2_000, 4_000_000),
        "gemini-2.0-flash-lite": TierRateLimits(4_000, 4_000_000),
        "gemini-1.5-flash": TierRateLimits(2_000, 4_000_000),
        "gemini-1.5-flash-8b": TierRateLimits(4_000, 4_000_000),
        "gemini-1.5-pro": TierRateLimits(1_000, 4_000_000),
    },
    APITier.TIER_2: {
        "gemini-2.5-flash-preview-05-20": TierRateLimits(2_000, 3_000_000),
        "gemini-2.5-pro-preview-06-05": TierRateLimits(1_000, 5_000_000),
        "gemini-2.0-flash": TierRateLimits(10_000, 10_000_000),
        "gemini-2.0-flash-lite": TierRateLimits(20_000, 10_000_000),
        "gemini-1.5-flash": TierRateLimits(2_000, 4_000_000),
        "gemini-1.5-flash-8b": TierRateLimits(4_000, 4_000_000),
        "gemini-1.5-pro": TierRateLimits(1_000, 4_000_000),
    },
    APITier.TIER_3: {
        "gemini-2.5-flash-preview-05-20": TierRateLimits(10_000, 8_000_000),
        "gemini-2.5-pro-preview-06-05": TierRateLimits(2_000, 8_000_000),
        "gemini-2.0-flash": TierRateLimits(30_000, 30_000_000),
        "gemini-2.0-flash-lite": TierRateLimits(30_000, 30_000_000),
    },
}

# Tier display names
TIER_NAMES = {
    APITier.FREE: "Free Tier",
    APITier.TIER_1: "Tier 1 (Billing Enabled)",
    APITier.TIER_2: "Tier 2",
    APITier.TIER_3: "Tier 3",
}

# Helper Functions


def _validate_api_key(api_key: str) -> None:
    """Validate API key format and raise appropriate errors"""
    if not api_key or not isinstance(api_key, str):
        raise ValueError("API key must be a non-empty string")

    api_key = api_key.strip()
    if len(api_key) < 30:
        raise ValueError("API key appears to be invalid (too short)")


def _parse_tier_from_string(tier_str: str) -> APITier | None:
    """Parse tier string to APITier enum with validation"""
    if not tier_str:
        return None

    tier_str = tier_str.lower().strip()
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


def _parse_enable_caching_from_env() -> bool:
    """Parse GEMINI_ENABLE_CACHING environment variable"""
    env_value = os.getenv("GEMINI_ENABLE_CACHING", "").lower().strip()
    return env_value in ("true", "1", "yes", "on")


# Smart Configuration Manager


class ConfigManager:
    """Handles the resolution of configuration options with a clear precedence:
    Explicit arguments > Environment variables > Sensible defaults.
    """

    def __init__(self, **overrides: Unpack[GeminiConfig]):
        # API Key Resolution
        self.api_key: str | None = overrides.get("api_key") or os.getenv(
            "GEMINI_API_KEY",
        )
        if self.api_key:
            _validate_api_key(self.api_key)
        else:
            raise ValueError(
                "GEMINI_API_KEY is required. You can provide it by:\n"
                "  • Setting GEMINI_API_KEY environment variable\n"
                "  • Using config_scope(api_key='your-key')\n"
                "  • Passing api_key= parameter directly",
            )

        # Tier Resolution
        tier_override = overrides.get("tier")
        env_tier_str = os.getenv("GEMINI_TIER")
        if isinstance(tier_override, str):
            self.tier = _parse_tier_from_string(tier_override) or APITier.FREE
        elif isinstance(tier_override, APITier):
            self.tier = tier_override
        else:
            env_tier = _parse_tier_from_string(env_tier_str) if env_tier_str else None
            self.tier = env_tier or APITier.FREE

        # Model Resolution
        self.model: str = (
            overrides.get("model")
            or os.getenv("GEMINI_MODEL")
            or self._get_default_model()
        )

        # Caching Resolution (correctly handles `enable_caching=False`)
        if "enable_caching" in overrides:
            self.enable_caching: bool = overrides["enable_caching"]
        else:
            self.enable_caching: bool = _parse_enable_caching_from_env()

        # Validate model and caching after resolution
        self._validate_model_for_tier()
        self._validate_caching_configuration()

    def _validate_model_for_tier(self) -> None:
        """Validate that the selected model is available in the selected tier"""
        if self.model and not self.get_model_limits(self.model):
            available_models = self._get_available_models()
            raise GeminiBatchError(
                f"Model '{self.model}' is not available in {self.tier.value} tier.\n"
                f"Available models: {', '.join(available_models)}\n"
                f"You can change tiers with: config_scope(tier='tier_1') or tier= parameter",
            )

    def _validate_caching_configuration(self) -> None:
        """Validate caching configuration is consistent"""
        if not self.enable_caching:
            return

        caching_support = self.can_use_caching(self.model, MIN_CACHING_THRESHOLD)
        supports_any = caching_support.get("supported")

        if not supports_any:
            log.warning(
                "Caching enabled but model %s doesn't support it. Disabling caching.",
                self.model,
            )
            self.enable_caching = False

    def __repr__(self) -> str:
        return (
            f"ConfigManager(model='{self.model}', tier={self.tier.value}, "
            f"caching={self.enable_caching}, api_key_set={bool(self.api_key)})"
        )

    def _get_default_model(self) -> str:
        """Get the best default model available for the current tier"""
        tier_models = self._get_available_models()
        if not tier_models:
            return "gemini-2.0-flash"  # Absolute fallback
        # Simple heuristic: prefer "pro" then "flash"
        if "gemini-2.5-pro-preview-06-05" in tier_models:
            return "gemini-2.5-pro-preview-06-05"
        if "gemini-2.5-flash-preview-05-20" in tier_models:
            return "gemini-2.5-flash-preview-05-20"
        return tier_models[0]

    def _get_available_models(self) -> list[str]:
        """Get list of available model names for the current tier"""
        tier_limits = TIER_RATE_LIMITS.get(self.tier)
        if not tier_limits:
            return []
        return list(tier_limits.keys())

    def get_model_limits(self, model: str) -> ModelTierLimits | None:
        """Get combined model capabilities and rate limits for a specific model"""
        model_caps = MODEL_CAPABILITIES.get(model)
        tier_limits = TIER_RATE_LIMITS.get(self.tier, {}).get(model)

        if not model_caps or not tier_limits:
            return None

        return ModelTierLimits(
            requests_per_minute=tier_limits.requests_per_minute,
            tokens_per_minute=tier_limits.tokens_per_minute,
            supports_multimodal=model_caps.supports_multimodal,
            context_window=model_caps.context_window,
            caching=model_caps.caching,
        )

    def select_optimal_model(
        self,
        content_tokens: int,
        query_count: int,
        video_count: int = 1,
    ) -> str:
        """Enhanced model selection - currently returns the default model"""
        return self.model

    def get_rate_limiter_config(self, model: str) -> dict[str, int]:
        """Get rate limiter configuration for a given model"""
        limits = self.get_model_limits(model)
        if not limits:
            available_models = self._get_available_models()
            raise GeminiBatchError(
                f"Model '{model}' is not available in {self.tier.value} tier. "
                f"Available models: {', '.join(available_models)}",
            )
        return {
            "requests_per_minute": limits.requests_per_minute,
            "tokens_per_minute": limits.tokens_per_minute,
        }

    def get_tier_name(self) -> str:
        """Get human-readable tier name"""
        return TIER_NAMES.get(self.tier, "Unknown Tier")

    def can_use_caching(
        self,
        model: str,
        token_count: int,
        prefer_implicit: bool = True,
    ) -> dict[str, bool | CachingRecommendation]:
        """Determine if caching is supported and recommended for a given context"""
        if not self.enable_caching:
            return {"supported": False, "recommendation": CachingRecommendation.NONE}

        limits = self.get_model_limits(model)
        if not limits or not limits.caching:
            return {"supported": False, "recommendation": CachingRecommendation.NONE}

        caching_caps = limits.caching
        can_implicit = caching_caps.can_use_implicit_caching(token_count)
        can_explicit = caching_caps.can_use_explicit_caching(token_count)

        recommendation = CachingRecommendation.NONE
        if can_implicit and prefer_implicit:
            recommendation = CachingRecommendation.IMPLICIT
        elif can_explicit:
            recommendation = CachingRecommendation.EXPLICIT
        elif can_implicit:  # Fallback if explicit not possible
            recommendation = CachingRecommendation.IMPLICIT

        return {
            "supported": can_implicit or can_explicit,
            "implicit": can_implicit,
            "explicit": can_explicit,
            "recommendation": recommendation,
        }

    def get_caching_thresholds(self, model: str) -> dict[str, int | None]:
        """Get the token thresholds for implicit and explicit caching for a model"""
        limits = self.get_model_limits(model)
        if not limits or not limits.caching:
            return {"implicit": None, "explicit": None}

        caching_caps = limits.caching
        return {
            "implicit": caching_caps.implicit_minimum_tokens,
            "explicit": caching_caps.explicit_minimum_tokens,
        }

    def validate_configuration(self) -> list[str]:
        """Validate entire configuration and return issues"""
        issues = []
        if not self.api_key:
            issues.append("Missing API key")
        if not self.get_model_limits(self.model):
            issues.append(f"Model {self.model} not available in {self.tier.value} tier")
        if self.enable_caching:
            caching_support = self.can_use_caching(self.model, MIN_CACHING_THRESHOLD)
            if not caching_support.get("supported"):
                issues.append(f"Caching enabled but not supported by {self.model}")
        return issues

    def get_config_summary(self) -> dict[str, Any]:
        """Get summary of current configuration for debugging"""
        return {
            "tier": self.tier.value,
            "tier_name": self.get_tier_name(),
            "model": self.model,
            "enable_caching": self.enable_caching,
            "api_key_present": bool(self.api_key),
        }


# Ambient Configuration (Thread-safe and async-safe)

# A thread-safe, async-safe container for the ambient configuration.
_ambient_config_var: contextvars.ContextVar[ConfigManager | None] = (
    contextvars.ContextVar("gemini_batch_config", default=None)
)


def get_config() -> ConfigManager:
    """Gets the configuration from the current context.
    If no configuration is set, it creates a default one from environment variables.
    """
    config = _ambient_config_var.get()
    if config is None:
        config = ConfigManager()
        _ambient_config_var.set(config)
    return config


@contextmanager
def config_scope(**config: Unpack[GeminiConfig]):
    """A context manager to temporarily use a different configuration.
    This is thread-safe and ideal for testing or specific overrides.

    Example:
        with config_scope(model="gemini-2.5-pro"):
            # Code inside this block will use gemini-2.5-pro
            processor = BatchProcessor()
    """
    token = _ambient_config_var.set(ConfigManager(**config))
    try:
        yield
    finally:
        _ambient_config_var.reset(token)


def get_effective_config(**overrides: Unpack[GeminiConfig]) -> dict[str, Any]:
    """Show what configuration would actually be used with these overrides.
    Useful for debugging configuration precedence.

    Example:
        # See what config would be used
        print(get_effective_config(model="gemini-2.5-pro"))

        # Compare with current ambient config
        print(get_effective_config())
    """
    base_config = get_config()
    effective = base_config.get_config_summary()

    # Apply overrides to show final result
    if overrides:
        temp_config = ConfigManager(
            **{
                "api_key": base_config.api_key,
                "model": base_config.model,
                "tier": base_config.tier,
                "enable_caching": base_config.enable_caching,
                **overrides,
            },
        )
        effective = temp_config.get_config_summary()

    # Add source information
    effective["_config_source"] = "ambient + overrides" if overrides else "ambient"
    return effective


def debug_config(**overrides: Unpack[GeminiConfig]) -> None:
    """Print current configuration for troubleshooting.

    Example:
        debug_config()  # Show current config
        debug_config(model="gemini-2.5-pro")  # Show config with overrides
    """
    effective = get_effective_config(**overrides)

    print("Gemini Batch Configuration:")
    print(f"  Model: {effective['model']}")
    print(f"  Tier: {effective['tier_name']}")
    print(f"  Caching: {effective['enable_caching']}")
    print(f"  API Key: {'✓ Set' if effective['api_key_present'] else '✗ Missing'}")
    if overrides:
        print(f"  Overrides: {overrides}")
    print(f"  Source: {effective['_config_source']}")
