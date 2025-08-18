"""Configuration audit and source tracking.

This module provides the SourceMap system for tracking where each configuration
value originated, with proper secret redaction for security.
"""

from typing import Any

from .types import ConfigOrigin, SourceMap


class SourceTracker:
    """Tracks the origin of configuration values during resolution.

    This class builds up a SourceMap as configuration is resolved from
    multiple sources, ensuring audit-grade tracking of where each value
    came from.
    """

    def __init__(self) -> None:
        """Initialize an empty source tracker."""
        self._origins: dict[str, ConfigOrigin] = {}

    def set_origin(self, field: str, origin: ConfigOrigin) -> None:
        """Record the origin of a configuration field.

        Args:
            field: The configuration field name
            origin: Where the value came from
        """
        self._origins[field] = origin

    def set_multiple(self, fields: dict[str, Any], origin: ConfigOrigin) -> None:
        """Record the origin for multiple fields at once.

        Args:
            fields: Dictionary of field names to values
            origin: Where all these values came from
        """
        for field in fields:
            self._origins[field] = origin

    def get_source_map(self) -> SourceMap:
        """Get the current source map.

        Returns:
            Immutable mapping of field names to their origins.
        """
        return dict(self._origins)  # Return a copy for immutability

    def has_origin(self, field: str) -> bool:
        """Check if we have an origin recorded for a field.

        Args:
            field: The field name to check

        Returns:
            True if origin is recorded, False otherwise.
        """
        return field in self._origins


def generate_telemetry_summary(source_map: SourceMap) -> dict[str, int]:
    """Generate telemetry summary from a source map.

    This creates aggregate counts suitable for metrics without revealing
    any sensitive configuration values.

    Args:
        source_map: The source map to summarize

    Returns:
        Dictionary with counts per origin type (e.g., {"env": 3, "file": 2})
    """
    counts: dict[str, int] = {}

    for origin in source_map.values():
        counts[origin] = counts.get(origin, 0) + 1

    return counts


def generate_redacted_audit(config_dict: dict[str, Any], source_map: SourceMap) -> str:
    """Generate a redacted audit report showing field origins.

    This creates a human-readable report that shows where each configuration
    value came from without revealing sensitive information.

    Args:
        config_dict: The configuration values
        source_map: The source origins for each field

    Returns:
        Redacted audit report as a string
    """
    # Sensitive fields that should be redacted
    sensitive_fields = {"api_key"}

    lines = []

    # Define a consistent field order for output
    field_order = [
        "api_key",
        "model",
        "tier",
        "enable_caching",
        "use_real_api",
        "ttl_seconds",
    ]

    for field in field_order:
        if field in source_map:
            origin = source_map[field]

            if field in sensitive_fields:
                # Redact sensitive fields
                if origin == "env":
                    value_display = f"env:GEMINI_{field.upper()}"
                elif origin == "file":
                    value_display = "file:<redacted>"
                elif origin == "programmatic":
                    value_display = "programmatic:<redacted>"
                else:
                    value_display = f"{origin}:<redacted>"
            else:
                # Show actual values for non-sensitive fields
                actual_value = config_dict.get(field, "<missing>")
                if origin == "env":
                    env_var = f"GEMINI_{field.upper()}"
                    value_display = f"env:{env_var}={actual_value}"
                elif origin == "file":
                    value_display = f"file:{actual_value}"
                else:
                    value_display = f"{origin}:{actual_value}"

            lines.append(f"{field}: {value_display}")

    return "\n".join(lines)
