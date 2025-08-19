"""Configuration management for the Gemini Batch Pipeline.

This module provides the new resolve-once, freeze-then-flow configuration system
that replaces the previous ambient context-based approach.

Key components:
- ResolvedConfig: Post-resolution configuration with audit metadata
- FrozenConfig: Immutable configuration for pipeline execution
- SourceMap: Audit tracking of configuration value origins
"""

# ruff: noqa: I001

# --- Configuration System Exports ---

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
# Compatibility and migration helpers removed

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
]
