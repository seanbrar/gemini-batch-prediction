"""Configuration compatibility utilities for migration.

This module provides utilities for working with both legacy GeminiConfig
(dict-based) and new FrozenConfig (dataclass) during the migration period.
"""

from typing import Any

from ..core.models import APITier
from .api import resolve_config
from .types import FrozenConfig

# Type alias for the union type used during migration
ConfigType = FrozenConfig | dict[str, Any]  # GeminiConfig is TypedDict -> dict


class ConfigCompatibilityShim:
    """Compatibility shim for accessing configuration fields.

    This class provides a unified interface for accessing configuration
    fields regardless of whether the underlying config is FrozenConfig
    or legacy GeminiConfig dict.
    """

    def __init__(self, config: ConfigType) -> None:
        """Initialize the compatibility shim.

        Args:
            config: Either a FrozenConfig or legacy GeminiConfig dict
        """
        self._config = config
        # Use more robust check for FrozenConfig that handles import path differences
        self._is_frozen = isinstance(config, FrozenConfig) or (
            hasattr(config, "__class__")
            and config.__class__.__name__ == "FrozenConfig"
            and "config.types" in config.__class__.__module__
        )

    @property
    def api_key(self) -> str | None:
        """Get the API key from either config type."""
        if self._is_frozen:
            return self._config.api_key
        return self._config.get("api_key")

    @property
    def model(self) -> str:
        """Get the model from either config type."""
        if self._is_frozen:
            return self._config.model
        return self._config.get("model", "gemini-2.0-flash")

    @property
    def tier(self) -> APITier:
        """Get the tier from either config type."""
        if self._is_frozen:
            return self._config.tier

        # Handle legacy dict format
        tier_value = self._config.get("tier", "free")
        if isinstance(tier_value, APITier):
            return tier_value

        # Convert string to APITier
        tier_map = {
            "free": APITier.FREE,
            "tier_1": APITier.TIER_1,
            "tier_2": APITier.TIER_2,
            "tier_3": APITier.TIER_3,
        }
        return tier_map.get(str(tier_value).lower(), APITier.FREE)

    @property
    def enable_caching(self) -> bool:
        """Get the caching setting from either config type."""
        if self._is_frozen:
            return self._config.enable_caching
        return self._config.get("enable_caching", True)

    @property
    def use_real_api(self) -> bool:
        """Get the real API setting from either config type."""
        if self._is_frozen:
            return self._config.use_real_api
        return self._config.get("use_real_api", False)

    @property
    def ttl_seconds(self) -> int:
        """Get the TTL setting from either config type."""
        if self._is_frozen:
            return self._config.ttl_seconds
        return self._config.get("ttl_seconds", 3600)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary format.

        This is useful for legacy code that expects dict access.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "api_key": self.api_key,
            "model": self.model,
            "tier": self.tier,
            "enable_caching": self.enable_caching,
            "use_real_api": self.use_real_api,
            "ttl_seconds": self.ttl_seconds,
        }

    def to_frozen(self) -> FrozenConfig:
        """Convert to FrozenConfig format.

        Returns:
            FrozenConfig representation of the configuration
        """
        if self._is_frozen:
            return self._config  # type: ignore[return-value]

        # Convert from dict to FrozenConfig
        return FrozenConfig(
            api_key=self.api_key,
            model=self.model,
            tier=self.tier,
            enable_caching=self.enable_caching,
            use_real_api=self.use_real_api,
            ttl_seconds=self.ttl_seconds,
        )

    @property
    def is_frozen_config(self) -> bool:
        """Check if the underlying config is FrozenConfig.

        Returns:
            True if using new FrozenConfig, False if legacy dict
        """
        return self._is_frozen


def ensure_frozen_config(config: ConfigType) -> FrozenConfig:
    """Ensure a configuration is in FrozenConfig format.

    This function converts legacy GeminiConfig dicts to FrozenConfig,
    or returns the FrozenConfig unchanged.

    Args:
        config: Either a FrozenConfig or legacy GeminiConfig dict

    Returns:
        FrozenConfig representation

    Example:
        # Works with both types
        frozen1 = ensure_frozen_config(some_frozen_config)
        frozen2 = ensure_frozen_config(some_dict_config)
    """
    if isinstance(config, FrozenConfig):
        return config

    # Convert from dict using the shim
    shim = ConfigCompatibilityShim(config)
    return shim.to_frozen()


def ensure_dict_config(config: ConfigType) -> dict[str, Any]:
    """Ensure a configuration is in dict format.

    This function converts FrozenConfig to dict format for legacy code,
    or returns the dict unchanged.

    Args:
        config: Either a FrozenConfig or legacy GeminiConfig dict

    Returns:
        Dictionary representation

    Example:
        # Works with both types
        dict1 = ensure_dict_config(some_frozen_config)
        dict2 = ensure_dict_config(some_dict_config)
    """
    if isinstance(config, dict):
        return config

    # Convert from FrozenConfig using the shim
    shim = ConfigCompatibilityShim(config)
    return shim.to_dict()


def migrate_config_to_frozen(
    config: dict[str, Any] | None = None, *, resolve_if_none: bool = True
) -> FrozenConfig:
    """Migrate a legacy config dict to FrozenConfig using the new resolution system.

    This is the recommended way to upgrade from legacy config usage.

    Args:
        config: Legacy config dict, or None to resolve from environment
        resolve_if_none: If True and config is None, resolve from environment

    Returns:
        FrozenConfig resolved through the new system

    Raises:
        ValueError: If config is None and resolve_if_none is False

    Example:
        # Migrate existing dict config
        legacy_config = {"model": "gemini-2.0-pro", "use_real_api": True}
        frozen = migrate_config_to_frozen(legacy_config)

        # Resolve from environment
        frozen = migrate_config_to_frozen()
    """
    if config is None:
        if not resolve_if_none:
            raise ValueError("Config is None and resolve_if_none is False")

        # Resolve using the new system
        resolved = resolve_config()
        return resolved.to_frozen()

    # Migrate the provided dict config
    resolved = resolve_config(programmatic=config)
    return resolved.to_frozen()


# Convenience functions for common migration patterns


def get_config_field(config: ConfigType, field: str, default: Any = None) -> Any:
    """Get a field value from either config type.

    Args:
        config: Either FrozenConfig or dict config
        field: Field name to retrieve
        default: Default value if field not found

    Returns:
        Field value or default

    Example:
        model = get_config_field(config, "model", "gemini-2.0-flash")
    """
    if isinstance(config, FrozenConfig):
        return getattr(config, field, default)
    return config.get(field, default)


def is_frozen_config(config: ConfigType) -> bool:
    """Check if a config is the new FrozenConfig type.

    Args:
        config: Configuration to check

    Returns:
        True if FrozenConfig, False if legacy dict
    """
    return isinstance(config, FrozenConfig)
