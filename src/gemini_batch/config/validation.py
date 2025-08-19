"""Enhanced validation rules and invariant checking.

This module provides additional validation rules beyond the basic Pydantic
schema validation, including complex invariants and business logic validation.
"""

from typing import Any

from ..core.models import APITier
from .schema import GeminiSettings
from .types import ResolvedConfig


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails beyond basic schema validation."""

    def __init__(
        self, field: str, value: Any, message: str, suggestion: str | None = None
    ) -> None:
        """Initialize configuration validation error.

        Args:
            field: The configuration field that failed validation
            value: The invalid value
            message: Human-readable error message
            suggestion: Optional suggestion for fixing the error
        """
        self.field = field
        self.value = value
        self.message = message
        self.suggestion = suggestion

        error_msg = f"Configuration validation failed for '{field}': {message}"
        if suggestion:
            error_msg += f"\nSuggestion: {suggestion}"

        super().__init__(error_msg)


def validate_resolved_config(config: ResolvedConfig) -> None:
    """Perform comprehensive validation on a resolved configuration.

    This function performs validation beyond what Pydantic can handle,
    including complex invariants and business logic validation.

    Args:
        config: The resolved configuration to validate

    Raises:
        ConfigValidationError: If validation fails

    Example:
        config = resolve_config()
        validate_resolved_config(config)  # Raises if invalid
    """
    # Validate API key requirement
    _validate_api_key_requirement(config)

    # Validate model and tier compatibility
    _validate_model_tier_compatibility(config)

    # Validate caching configuration
    _validate_caching_configuration(config)

    # Validate TTL settings
    _validate_ttl_settings(config)


def validate_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Validate a configuration dictionary and return normalized values.

    This function performs validation on raw configuration data before
    it's converted to a ResolvedConfig.

    Args:
        config_dict: Raw configuration dictionary

    Returns:
        Validated and normalized configuration dictionary

    Raises:
        ConfigValidationError: If validation fails
    """
    # Use Pydantic for basic validation and normalization
    try:
        settings = GeminiSettings(**config_dict)
        validated_dict = settings.to_dict()
    except Exception as e:
        raise ConfigValidationError(
            field="<multiple>",
            value=config_dict,
            message=f"Schema validation failed: {e}",
            suggestion="Check field types and required values",
        ) from e

    # Additional validation on the validated dict
    _validate_field_combinations(validated_dict)

    return validated_dict


def _validate_api_key_requirement(config: ResolvedConfig) -> None:
    """Validate API key requirement based on use_real_api setting.

    Note: This validation is also performed by the Pydantic schema,
    but kept here for additional business logic validation if needed.
    """
    # This validation is primarily handled by the Pydantic schema
    # in schema.py:97-106. Keeping this as a placeholder for any
    # additional business logic validation that might be needed.


def _validate_model_tier_compatibility(config: ResolvedConfig) -> None:
    """Validate that the model is compatible with the selected tier."""
    # The tier should already be validated by Pydantic as an APITier enum
    # This function is for additional business logic validation

    # For now, all tiers are valid for all models
    # Future: Add model-specific tier requirements here


def _validate_caching_configuration(config: ResolvedConfig) -> None:
    """Validate caching-related configuration."""
    if config.enable_caching and config.ttl_seconds <= 0:
        raise ConfigValidationError(
            field="ttl_seconds",
            value=config.ttl_seconds,
            message="TTL must be positive when caching is enabled",
            suggestion="Set ttl_seconds to a positive value or disable caching",
        )


def _validate_ttl_settings(config: ResolvedConfig) -> None:
    """Validate TTL (Time To Live) settings.

    Note: Basic TTL validation (>= 1) is handled by the Pydantic schema.
    """
    # Basic validation (ttl_seconds >= 1) is handled by Pydantic schema
    # in schema.py:65 with ge=1 constraint.

    # Keep additional business logic validations here
    # Warn about very long TTLs (more than 24 hours)
    if config.ttl_seconds > 86400:  # 24 hours
        # This is a warning, not an error - we don't raise here
        # In a production system, this might be logged
        pass


def _validate_field_combinations(config_dict: dict[str, Any]) -> None:
    """Validate combinations of fields that have dependencies."""
    # Example: if using a specific model, certain features might be required
    model = config_dict.get("model", "")  # noqa: F841
    tier = config_dict.get("tier")  # noqa: F841

    # For now, all combinations are valid
    # Future: Add specific model/tier compatibility rules here


def check_config_security(config: ResolvedConfig) -> list[str]:
    """Check configuration for potential security issues.

    Args:
        config: The resolved configuration to check

    Returns:
        List of security warnings (empty if no issues found)

    Example:
        config = resolve_config()
        warnings = check_config_security(config)
        for warning in warnings:
            print(f"Security warning: {warning}")
    """
    warnings = []

    # Check for missing API key in real API mode
    if config.use_real_api and not config.api_key:
        warnings.append("Real API mode enabled but no API key provided")

    # Check for very long TTL values
    if config.ttl_seconds > 86400:  # 24 hours
        warnings.append(f"Very long TTL configured: {config.ttl_seconds} seconds")

    # Check for development settings in production-like configurations
    if config.use_real_api and config.ttl_seconds < 60:
        warnings.append("Short TTL with real API may cause excessive requests")

    return warnings


def suggest_config_improvements(config: ResolvedConfig) -> list[str]:
    """Suggest potential improvements to the configuration.

    Args:
        config: The resolved configuration to analyze

    Returns:
        List of improvement suggestions

    Example:
        config = resolve_config()
        suggestions = suggest_config_improvements(config)
        for suggestion in suggestions:
            print(f"Suggestion: {suggestion}")
    """
    suggestions = []

    # Suggest enabling caching if disabled
    if not config.enable_caching:
        suggestions.append("Enable caching for better performance and cost savings")

    # Suggest appropriate TTL values
    if config.enable_caching and config.ttl_seconds < 3600:
        suggestions.append(
            "Consider increasing TTL to at least 1 hour for better cache efficiency"
        )

    # Suggest tier upgrades if using free tier
    if config.tier == APITier.FREE and config.use_real_api:
        suggestions.append("Consider upgrading to a paid tier for higher rate limits")

    return suggestions
