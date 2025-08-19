"""Configuration management for the Gemini Batch Pipeline.

This module provides the new resolve-once, freeze-then-flow configuration system
that replaces the previous ambient context-based approach.

Key components:
- ResolvedConfig: Post-resolution configuration with audit metadata
- FrozenConfig: Immutable configuration for pipeline execution
- SourceMap: Audit tracking of configuration value origins
"""

# ruff: noqa: I001

# --- New Configuration System ---
from collections.abc import Generator
from contextlib import contextmanager
import contextvars
import os

# --- Legacy Compatibility (temporary) ---
# Import the old config system to maintain compatibility during migration
import sys
from typing import TypedDict

from gemini_batch.core.models import APITier

from .api import (
    check_environment,
    get_effective_profile,
    list_available_profiles,
    print_config_audit,
    print_effective_config,
    resolve_config,
    validate_profile,
)
from .scope import config_override, config_scope, get_ambient_resolved_config
from .audit import SourceTracker, generate_redacted_audit, generate_telemetry_summary
from .file_loader import ConfigFileError, FileConfigLoader
from .providers import (
    GeminiConfigBuilder,
    GeminiProviderConfig,
    ProviderConfigBuilder,
    ProviderConfigRegistry,
    build_provider_config,
    get_provider_registry,
    list_registered_providers,
    register_provider,
)
from .resolver import ConfigResolver
from .schema import GeminiSettings
from .telemetry import (
    ConfigTelemetryMixin,
    get_config_telemetry_summary,
    record_config_file_load,
    record_config_provider_build,
    record_config_resolution,
    record_config_scope_usage,
    record_config_validation_error,
)
from .types import ConfigOrigin, FrozenConfig, ResolvedConfig, SourceMap
from .validation import (
    ConfigValidationError,
    check_config_security,
    suggest_config_improvements,
    validate_config_dict,
    validate_resolved_config,
)
from .compatibility import (
    ConfigCompatibilityShim,
    ensure_frozen_config,
    ensure_dict_config,
    migrate_config_to_frozen,
    get_config_field,
    is_frozen_config,
)


# Legacy GeminiConfig type (from the old config.py)
class GeminiConfig(TypedDict, total=False):
    """Legacy configuration type for backward compatibility."""

    api_key: str
    model: str
    tier: str | APITier
    enable_caching: bool
    use_real_api: bool
    ttl_seconds: int


# Legacy ambient config system
_ambient_config_var: contextvars.ContextVar[GeminiConfig] = contextvars.ContextVar(
    "gemini_batch_config"
)


def get_ambient_config() -> GeminiConfig:
    """Legacy ambient config resolution for backward compatibility."""
    try:
        return _ambient_config_var.get()
    except LookupError:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set in the environment and no explicit "
                "config was provided."
            ) from None

        return GeminiConfig(
            api_key=api_key,
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        )


@contextmanager
def legacy_config_scope(config: GeminiConfig) -> Generator[None]:
    """Legacy config scope for backward compatibility.

    DEPRECATED: Use the new config_scope() with ResolvedConfig instead.
    This function is kept for backward compatibility during migration.
    """
    token = _ambient_config_var.set(config)
    try:
        yield
    finally:
        _ambient_config_var.reset(token)


# For backward compatibility, alias the legacy function to config_scope
# This allows existing code to continue working during migration
_legacy_config_scope = legacy_config_scope


__all__ = [  # noqa: RUF022
    # New system - main API
    "resolve_config",
    "list_available_profiles",
    "get_effective_profile",
    "validate_profile",
    "print_config_audit",
    "print_effective_config",
    "check_environment",
    # Scoping
    "config_scope",
    "config_override",
    "get_ambient_resolved_config",
    # Core types
    "ResolvedConfig",
    "FrozenConfig",
    "SourceMap",
    "ConfigOrigin",
    # Advanced usage
    "GeminiSettings",
    "ConfigResolver",
    "FileConfigLoader",
    "ConfigFileError",
    "SourceTracker",
    "generate_redacted_audit",
    "generate_telemetry_summary",
    # Providers
    "ProviderConfigBuilder",
    "ProviderConfigRegistry",
    "register_provider",
    "build_provider_config",
    "get_provider_registry",
    "list_registered_providers",
    "GeminiProviderConfig",
    "GeminiConfigBuilder",
    # Telemetry
    "record_config_resolution",
    "record_config_validation_error",
    "record_config_scope_usage",
    "record_config_file_load",
    "record_config_provider_build",
    "ConfigTelemetryMixin",
    "get_config_telemetry_summary",
    # Validation
    "validate_resolved_config",
    "validate_config_dict",
    "ConfigValidationError",
    "check_config_security",
    "suggest_config_improvements",
    # Compatibility
    "ConfigCompatibilityShim",
    "ensure_frozen_config",
    "ensure_dict_config",
    "migrate_config_to_frozen",
    "get_config_field",
    "is_frozen_config",
    # Legacy compatibility
    "GeminiConfig",
    "get_ambient_config",
    "legacy_config_scope",
]
