"""
Configuration system for rate limiting and model capabilities
"""  # noqa: D200, D212, D415

from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
from enum import Enum
import logging
import os

# Use typing_extensions for Unpack for Python < 3.11
from typing import Any, Dict, List, Optional, Protocol, TypedDict, Union, Unpack  # noqa: UP035

from gemini_batch.constants import MIN_CACHING_THRESHOLD

from .exceptions import GeminiBatchError

log = logging.getLogger(__name__)


# Protocols & Type-Safe Configuration


class APITier(Enum):  # noqa: D101
    FREE = "free"
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


class GeminiConfig(TypedDict, total=False):
    """
    Type-safe configuration options for the Gemini client and processor.
    All parameters are optional and will fall back to environment variables or defaults.
    """  # noqa: D205, D212

    api_key: str
    model: str
    tier: Union[str, APITier]  # noqa: UP007
    enable_caching: bool


class ConversationConfig(TypedDict, total=False):
    """Additional, type-safe options specific to conversation sessions."""

    max_history_turns: int


class ClientProtocol(Protocol):
    """Protocol defining the essential capabilities of a content generation client."""

    def generate_content(self, content: Any, prompt: str, **kwargs) -> Any: ...  # noqa: ANN003, ANN401, D102
    def generate_batch(self, content: Any, questions: List[str], **kwargs) -> Any: ...  # noqa: ANN003, ANN401, D102, UP006


class ProcessorProtocol(Protocol):
    """Protocol defining the essential capabilities of a question processor."""

    def process_questions(  # noqa: D102
        self, content: Any, questions: List[str], **kwargs  # noqa: ANN003, ANN401, COM812, UP006
    ) -> dict: ...


# Data models for capabilities and limits


class CachingRecommendation(Enum):
    """Recommended caching strategy for a given context"""  # noqa: D415

    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
    NONE = "none"


@dataclass
class CachingCapabilities:
    """Caching capabilities for a model"""  # noqa: D415

    supports_implicit: bool = False
    supports_explicit: bool = True  # Default to explicit for all caching-enabled models
    implicit_minimum_tokens: Optional[int] = None  # noqa: UP045
    explicit_minimum_tokens: int = 4096

    def can_use_implicit_caching(self, token_count: int) -> bool:
        """Check if implicit caching is available for given token count"""  # noqa: D415
        return (
            self.supports_implicit
            and self.implicit_minimum_tokens is not None
            and token_count >= self.implicit_minimum_tokens
        )

    def can_use_explicit_caching(self, token_count: int) -> bool:
        """Check if explicit caching is available for given token count"""  # noqa: D415
        return self.supports_explicit and token_count >= self.explicit_minimum_tokens


@dataclass
class ModelCapabilities:
    """Intrinsic model capabilities (same across all tiers)"""  # noqa: D415

    supports_multimodal: bool
    context_window: int
    caching: Optional[CachingCapabilities] = None  # noqa: UP045

    def get_caching(self) -> CachingCapabilities:
        """Get caching capabilities with fallback"""  # noqa: D415
        if self.caching is not None:
            return self.caching
        return CachingCapabilities(
            supports_implicit=False,
            supports_explicit=False,
            explicit_minimum_tokens=float("inf"),
        )


@dataclass
class TierRateLimits:
    """Rate limits for a specific model in a specific tier"""  # noqa: D415

    requests_per_minute: int
    tokens_per_minute: int


@dataclass
class ModelTierLimits:
    """Combined model capabilities and rate limits for a specific tier"""  # noqa: D415

    requests_per_minute: int
    tokens_per_minute: int
    supports_multimodal: bool
    context_window: int
    caching: Optional[CachingCapabilities] = None  # noqa: UP045


def _explicit_only_caching() -> CachingCapabilities:
    """Standard explicit-only caching configuration"""  # noqa: D415
    return CachingCapabilities(
        supports_implicit=False,
        supports_explicit=True,
        explicit_minimum_tokens=4096,
    )


def _gemini_25_flash_caching() -> CachingCapabilities:
    """Gemini 2.5 Flash caching configuration with implicit support"""  # noqa: D415
    return CachingCapabilities(
        supports_implicit=True,
        supports_explicit=True,
        implicit_minimum_tokens=2048,
        explicit_minimum_tokens=4096,
    )


def _gemini_25_pro_caching() -> CachingCapabilities:
    """Gemini 2.5 Pro caching configuration with implicit support"""  # noqa: D415
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
    """Validate API key format and raise appropriate errors"""  # noqa: D415
    if not api_key or not isinstance(api_key, str):
        raise ValueError("API key must be a non-empty string")  # noqa: EM101, TRY003

    api_key = api_key.strip()
    if len(api_key) < 30:  # noqa: PLR2004
        raise ValueError("API key appears to be invalid (too short)")  # noqa: EM101, TRY003


def _parse_tier_from_string(tier_str: str) -> Optional[APITier]:  # noqa: UP045
    """Parse tier string to APITier enum with validation"""  # noqa: D415
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
    """Parse GEMINI_ENABLE_CACHING environment variable"""  # noqa: D415
    env_value = os.getenv("GEMINI_ENABLE_CACHING", "").lower().strip()
    return env_value in ("true", "1", "yes", "on")


# Smart Configuration Manager


class ConfigManager:
    """
    Handles the resolution of configuration options with a clear precedence:
    Explicit arguments > Environment variables > Sensible defaults.
    """  # noqa: D205, D212

    def __init__(self, **overrides: Unpack[GeminiConfig]):  # noqa: ANN204, D107
        # API Key Resolution
        self.api_key: Optional[str] = overrides.get("api_key") or os.getenv(  # noqa: UP045
            "GEMINI_API_KEY"  # noqa: COM812
        )
        if self.api_key:
            _validate_api_key(self.api_key)
        else:
            raise ValueError(  # noqa: TRY003
                "GEMINI_API_KEY is required. You can provide it by:\n"  # noqa: EM101
                "  • Setting GEMINI_API_KEY environment variable\n"
                "  • Using config_scope(api_key='your-key')\n"
                "  • Passing api_key= parameter directly"  # noqa: COM812
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
        """Validate that the selected model is available in the selected tier"""  # noqa: D415
        if self.model and not self.get_model_limits(self.model):
            available_models = self._get_available_models()
            raise GeminiBatchError(  # noqa: TRY003
                f"Model '{self.model}' is not available in {self.tier.value} tier.\n"  # noqa: EM102
                f"Available models: {', '.join(available_models)}\n"
                f"You can change tiers with: config_scope(tier='tier_1') or tier= parameter"  # noqa: COM812, E501
            )

    def _validate_caching_configuration(self) -> None:
        """Validate caching configuration is consistent"""  # noqa: D415
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

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ConfigManager(model='{self.model}', tier={self.tier.value}, "
            f"caching={self.enable_caching}, api_key_set={bool(self.api_key)})"
        )

    def _get_default_model(self) -> str:
        """Get the best default model available for the current tier"""  # noqa: D415
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
        """Get list of available model names for the current tier"""  # noqa: D415
        tier_limits = TIER_RATE_LIMITS.get(self.tier)
        if not tier_limits:
            return []
        return list(tier_limits.keys())

    def get_model_limits(self, model: str) -> Optional[ModelTierLimits]:  # noqa: UP045
        """Get combined model capabilities and rate limits for a specific model"""  # noqa: D415
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
        self, content_tokens: int, query_count: int, video_count: int = 1  # noqa: ARG002, COM812
    ) -> str:
        """Enhanced model selection - currently returns the default model"""  # noqa: D415
        return self.model

    def get_rate_limiter_config(self, model: str) -> Dict[str, int]:  # noqa: UP006
        """Get rate limiter configuration for a given model"""  # noqa: D415
        limits = self.get_model_limits(model)
        if not limits:
            available_models = self._get_available_models()
            raise GeminiBatchError(  # noqa: TRY003
                f"Model '{model}' is not available in {self.tier.value} tier. "  # noqa: EM102
                f"Available models: {', '.join(available_models)}"  # noqa: COM812
            )
        return {
            "requests_per_minute": limits.requests_per_minute,
            "tokens_per_minute": limits.tokens_per_minute,
        }

    def get_tier_name(self) -> str:
        """Get human-readable tier name"""  # noqa: D415
        return TIER_NAMES.get(self.tier, "Unknown Tier")

    def can_use_caching(
        self, model: str, token_count: int, prefer_implicit: bool = True  # noqa: COM812, FBT001, FBT002
    ) -> Dict[str, Union[bool, CachingRecommendation]]:  # noqa: UP006, UP007
        """Determine if caching is supported and recommended for a given context"""  # noqa: D415
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

    def get_caching_thresholds(self, model: str) -> Dict[str, Optional[int]]:  # noqa: UP006, UP045
        """Get the token thresholds for implicit and explicit caching for a model"""  # noqa: D415
        limits = self.get_model_limits(model)
        if not limits or not limits.caching:
            return {"implicit": None, "explicit": None}

        caching_caps = limits.caching
        return {
            "implicit": caching_caps.implicit_minimum_tokens,
            "explicit": caching_caps.explicit_minimum_tokens,
        }

    def validate_configuration(self) -> list[str]:
        """Validate entire configuration and return issues"""  # noqa: D415
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

    def get_config_summary(self) -> Dict[str, Any]:  # noqa: UP006
        """Get summary of current configuration for debugging"""  # noqa: D415
        return {
            "tier": self.tier.value,
            "tier_name": self.get_tier_name(),
            "model": self.model,
            "enable_caching": self.enable_caching,
            "api_key_present": bool(self.api_key),
        }


# Ambient Configuration (Thread-safe and async-safe)

# A thread-safe, async-safe container for the ambient configuration.
_ambient_config_var: contextvars.ContextVar[Optional[ConfigManager]] = (  # noqa: UP045
    contextvars.ContextVar("gemini_batch_config", default=None)
)


def get_config() -> ConfigManager:
    """
    Gets the configuration from the current context.
    If no configuration is set, it creates a default one from environment variables.
    """  # noqa: D205, D212
    config = _ambient_config_var.get()
    if config is None:
        config = ConfigManager()
        _ambient_config_var.set(config)
    return config


@contextmanager
def config_scope(**config: Unpack[GeminiConfig]):  # noqa: ANN201
    """
    A context manager to temporarily use a different configuration.
    This is thread-safe and ideal for testing or specific overrides.

    Example:
        with config_scope(model="gemini-2.5-pro"):
            # Code inside this block will use gemini-2.5-pro
            processor = BatchProcessor()
    """  # noqa: D205, D212
    token = _ambient_config_var.set(ConfigManager(**config))
    try:
        yield
    finally:
        _ambient_config_var.reset(token)


def get_effective_config(**overrides: Unpack[GeminiConfig]) -> Dict[str, Any]:  # noqa: UP006
    """
    Show what configuration would actually be used with these overrides.
    Useful for debugging configuration precedence.

    Example:
        # See what config would be used
        print(get_effective_config(model="gemini-2.5-pro"))

        # Compare with current ambient config
        print(get_effective_config())
    """  # noqa: D205, D212
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
            }  # noqa: COM812
        )
        effective = temp_config.get_config_summary()

    # Add source information
    effective["_config_source"] = "ambient + overrides" if overrides else "ambient"
    return effective


def debug_config(**overrides: Unpack[GeminiConfig]) -> None:
    """
    Print current configuration for troubleshooting.

    Example:
        debug_config()  # Show current config
        debug_config(model="gemini-2.5-pro")  # Show config with overrides
    """  # noqa: D212
    effective = get_effective_config(**overrides)

    print("Gemini Batch Configuration:")  # noqa: T201
    print(f"  Model: {effective['model']}")  # noqa: T201
    print(f"  Tier: {effective['tier_name']}")  # noqa: T201
    print(f"  Caching: {effective['enable_caching']}")  # noqa: T201
    print(f"  API Key: {'✓ Set' if effective['api_key_present'] else '✗ Missing'}")  # noqa: T201
    if overrides:
        print(f"  Overrides: {overrides}")  # noqa: T201
    print(f"  Source: {effective['_config_source']}")  # noqa: T201
