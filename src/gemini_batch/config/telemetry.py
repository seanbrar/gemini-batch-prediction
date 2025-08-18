"""Configuration telemetry integration.

This module integrates configuration events with the central telemetry system,
providing observability for configuration resolution, validation, and usage
patterns while maintaining the ultra-low overhead design.
"""

import hashlib
from typing import Any

from ..telemetry import TelemetryContextProtocol
from .audit import generate_telemetry_summary
from .types import ResolvedConfig


def record_config_resolution(
    tele: TelemetryContextProtocol,
    config: ResolvedConfig,
    profile: str | None = None,
) -> None:
    """Record configuration resolution metrics.

    Args:
        tele: The telemetry context to use
        config: The resolved configuration
        profile: The profile used, if any

    This function emits metrics about configuration resolution including
    source distribution and non-sensitive metadata.

    Example:
        with tele("config.resolution") as ctx:
            config = resolve_config(profile="production")
            record_config_resolution(ctx, config, profile="production")
    """
    # Generate source distribution metrics
    source_counts = generate_telemetry_summary(config.origin)

    # Record overall resolution event
    tele.count("resolution", 1, profile=profile or "default")

    # Record source distribution
    for source, count in source_counts.items():
        tele.gauge(f"sources.{source}", count, profile=profile or "default")

    # Record non-sensitive configuration fingerprint for change detection
    config_hash = _generate_config_hash(config)
    tele.metric("fingerprint", config_hash, profile=profile or "default")

    # Record field count
    tele.gauge("fields.total", len(config.origin), profile=profile or "default")


def record_config_validation_error(
    tele: TelemetryContextProtocol,
    error: Exception,
    config_fields: list[str],
) -> None:
    """Record configuration validation error metrics.

    Args:
        tele: The telemetry context to use
        error: The validation error that occurred
        config_fields: List of field names that were being validated

    Example:
        with tele("config.validation") as ctx:
            try:
                resolve_config({"use_real_api": True})  # Missing api_key
            except ValueError as e:
                record_config_validation_error(ctx, e, ["use_real_api"])
    """
    tele.count(
        "validation.errors",
        1,
        error_type=type(error).__name__,
        field_count=len(config_fields),
    )


def record_config_scope_usage(
    tele: TelemetryContextProtocol,
    scope_type: str,
    fields_overridden: int,
) -> None:
    """Record configuration scope usage metrics.

    Args:
        tele: The telemetry context to use
        scope_type: Type of scope used ("config_scope" or "config_override")
        fields_overridden: Number of fields overridden in the scope

    Example:
        with tele("config.scoping") as ctx:
            with config_override(model="gemini-2.0-pro"):
                record_config_scope_usage(ctx, "config_override", 1)
    """
    tele.count(
        "scope.usage", 1, scope_type=scope_type, fields_overridden=fields_overridden
    )


def record_config_file_load(
    tele: TelemetryContextProtocol,
    file_type: str,
    *,
    success: bool,
    profile: str | None = None,
) -> None:
    """Record configuration file loading metrics.

    Args:
        tele: The telemetry context to use
        file_type: Type of file ("project" or "home")
        success: Whether the file was loaded successfully
        profile: The profile used, if any

    Example:
        with tele("config.files") as ctx:
            try:
                config = load_project_config(profile="dev")
                record_config_file_load(ctx, "project", success=True, profile="dev")
            except ConfigFileError:
                record_config_file_load(ctx, "project", success=False, profile="dev")
    """
    result = "success" if success else "error"
    tele.count(f"files.{file_type}.{result}", 1, profile=profile or "default")


def record_config_provider_build(
    tele: TelemetryContextProtocol,
    provider: str,
    *,
    success: bool,
) -> None:
    """Record provider configuration building metrics.

    Args:
        tele: The telemetry context to use
        provider: The provider name
        success: Whether the provider config was built successfully

    Example:
        with tele("config.providers") as ctx:
            try:
                gemini_config = build_provider_config("gemini", frozen_config)
                record_config_provider_build(ctx, "gemini", success=True)
            except ValueError:
                record_config_provider_build(ctx, "gemini", success=False)
    """
    result = "success" if success else "error"
    tele.count(f"providers.{provider}.{result}", 1)


def _generate_config_hash(config: ResolvedConfig) -> str:
    """Generate a non-sensitive hash of the configuration.

    This hash can be used for correlation and change detection without
    revealing sensitive configuration values.

    Args:
        config: The resolved configuration

    Returns:
        Hex string hash of non-sensitive configuration data
    """
    # Include only non-sensitive fields in the hash
    hash_data = {
        "model": config.model,
        "tier": str(config.tier),
        "enable_caching": config.enable_caching,
        "use_real_api": config.use_real_api,
        "ttl_seconds": config.ttl_seconds,
        # Include source distribution for change detection
        "sources": dict(config.origin),
    }

    # Create deterministic hash
    hash_string = str(sorted(hash_data.items()))
    return hashlib.sha256(hash_string.encode()).hexdigest()[:16]


class ConfigTelemetryMixin:
    """Mixin to add telemetry integration to configuration classes.

    This mixin provides convenient methods for emitting configuration-related
    telemetry events from within configuration classes.
    """

    def _emit_resolution_telemetry(
        self,
        tele: TelemetryContextProtocol,
        config: ResolvedConfig,
        profile: str | None = None,
    ) -> None:
        """Emit telemetry for configuration resolution."""
        record_config_resolution(tele, config, profile)

    def _emit_validation_error_telemetry(
        self,
        tele: TelemetryContextProtocol,
        error: Exception,
        config_fields: list[str],
    ) -> None:
        """Emit telemetry for validation errors."""
        record_config_validation_error(tele, error, config_fields)

    def _emit_file_load_telemetry(
        self,
        tele: TelemetryContextProtocol,
        file_type: str,
        *,
        success: bool,
        profile: str | None = None,
    ) -> None:
        """Emit telemetry for file loading."""
        record_config_file_load(tele, file_type, success=success, profile=profile)


def get_config_telemetry_summary(
    source_counts: dict[str, int],
    total_resolutions: int = 1,
) -> dict[str, Any]:
    """Generate a summary of configuration telemetry data.

    Args:
        source_counts: Counts of fields by source type
        total_resolutions: Total number of resolutions

    Returns:
        Dictionary containing configuration telemetry summary

    Example:
        source_counts = {"env": 2, "file": 3, "default": 1}
        summary = get_config_telemetry_summary(source_counts)
        print(f"Configuration diversity: {summary['source_diversity']}")
    """
    total_fields = sum(source_counts.values())

    return {
        "total_fields": total_fields,
        "total_resolutions": total_resolutions,
        "source_distribution": source_counts,
        "source_diversity": len(source_counts),  # Number of different sources used
        "env_field_ratio": source_counts.get("env", 0) / total_fields
        if total_fields
        else 0,
        "file_field_ratio": source_counts.get("file", 0) / total_fields
        if total_fields
        else 0,
        "default_field_ratio": source_counts.get("default", 0) / total_fields
        if total_fields
        else 0,
        "programmatic_field_ratio": source_counts.get("programmatic", 0) / total_fields
        if total_fields
        else 0,
    }
