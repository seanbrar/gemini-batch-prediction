"""
Client configuration handling for Gemini API integration
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import warnings

from ..config import APITier, ConfigManager
from ..constants import RATE_LIMIT_WINDOW
from ..exceptions import MissingKeyError


@dataclass
class ClientConfiguration:
    """Configuration for GeminiClient initialization"""

    api_key: str
    model: str
    tier: APITier
    enable_caching: bool = False

    @property
    def model_name(self) -> str:
        """Compatibility property for GeminiClient"""
        warnings.warn(
            "The 'model_name' property is deprecated. Use 'model' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.model

    def validate(self) -> None:
        """Validate configuration has required values"""
        if not self.api_key:
            raise MissingKeyError(
                "API key required. Provide via parameter, ConfigManager, "
                "or GEMINI_API_KEY environment variable."
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for debugging and inspection"""
        return asdict(self)

    @classmethod
    def from_parameters(
        cls,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None,
        enable_caching: Optional[bool] = None,
        config_manager: Optional[ConfigManager] = None,
    ) -> "ClientConfiguration":
        """Create configuration from parameters with ConfigManager fallback"""
        config = config_manager or ConfigManager.from_env()

        return cls(
            api_key=api_key or config.api_key,
            model=model or config.model,
            tier=tier or config.tier,
            enable_caching=enable_caching
            if enable_caching is not None
            else config.enable_caching,
        )

    @classmethod
    def from_config_manager(
        cls, config_manager: ConfigManager, **overrides
    ) -> "ClientConfiguration":
        """Create configuration from ConfigManager with optional parameter overrides"""
        return cls(
            api_key=overrides.get("api_key") or config_manager.api_key,
            model=overrides.get("model") or config_manager.model,
            tier=overrides.get("tier") or config_manager.tier,
            enable_caching=overrides.get(
                "enable_caching", config_manager.enable_caching
            ),
        )


@dataclass
class RateLimitConfig:
    """Rate limiting parameters for API request throttling"""

    requests_per_minute: int
    tokens_per_minute: int
    window_seconds: int = RATE_LIMIT_WINDOW
