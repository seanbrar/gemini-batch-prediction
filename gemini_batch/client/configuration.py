"""
Client configuration classes for API connection and settings
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from ..config import APITier, ConfigManager
from ..constants import RATE_LIMIT_WINDOW
from ..exceptions import MissingKeyError


@dataclass
class ClientConfiguration:
    """Client configuration for API connection and model settings"""

    api_key: str
    model_name: str
    enable_caching: bool = False
    tier: Optional[APITier] = None
    custom_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config_manager(
        cls, config_manager: ConfigManager, **overrides
    ) -> "ClientConfiguration":
        """Create configuration from ConfigManager with optional parameter overrides"""
        return cls(
            api_key=overrides.get("api_key") or config_manager.api_key,
            model_name=overrides.get("model_name") or config_manager.model,
            tier=overrides.get("tier"),
            enable_caching=overrides.get("enable_caching", False),
            custom_options=overrides.get("custom_options", {}),
        )

    @classmethod
    def from_parameters(
        cls,
        api_key: str = None,
        model_name: str = None,
        tier: Optional[APITier] = None,
        **kwargs,
    ) -> "ClientConfiguration":
        """Create configuration from individual parameters with defaults"""
        # Create ConfigManager to handle defaults and environment
        config_manager = ConfigManager(tier=tier, model=model_name, api_key=api_key)

        return cls(
            api_key=api_key or config_manager.api_key,
            model_name=model_name or config_manager.model,
            tier=tier,
            enable_caching=kwargs.get("enable_caching", False),
            custom_options={k: v for k, v in kwargs.items() if k != "enable_caching"},
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for debugging and inspection"""
        return asdict(self)

    def validate(self) -> None:
        """Validate configuration has required values"""
        if not self.api_key:
            raise MissingKeyError(
                "API key required. Provide via parameter, ConfigManager, "
                "or GEMINI_API_KEY environment variable."
            )


@dataclass
class RateLimitConfig:
    """Rate limiting parameters for API request throttling"""

    requests_per_minute: int
    tokens_per_minute: int
    window_seconds: int = RATE_LIMIT_WINDOW
