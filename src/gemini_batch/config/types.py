"""Core configuration data types for the gemini_batch pipeline.

This module defines the fundamental data structures used throughout the configuration
system, following the resolve-once, freeze-then-flow pattern.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, NamedTuple

from gemini_batch.core.models import APITier

# --- Source Tracking Types ---

ConfigOrigin = Literal["programmatic", "env", "file", "default"]
SourceMap = Mapping[str, ConfigOrigin]

# --- Core Configuration Data ---


class ResolvedConfig(NamedTuple):
    """Configuration after resolution from all sources, before freezing.

    This represents the validated, merged result of combining programmatic overrides,
    environment variables, files, and defaults. It includes audit metadata for
    observability.

    The ResolvedConfig is logically immutable but provides helper methods for
    creating variants and converting to the frozen pipeline format.
    """

    # Core fields from the specification
    api_key: str | None
    model: str
    tier: APITier
    enable_caching: bool
    use_real_api: bool
    ttl_seconds: int

    # Audit metadata - tracks where each field value came from
    origin: SourceMap

    def __str__(self) -> str:
        """String representation with redacted API key for safe logging."""
        api_key_display = "[REDACTED]" if self.api_key else None
        return (
            f"ResolvedConfig(api_key={api_key_display!r}, model={self.model!r}, "
            f"tier={self.tier!r}, enable_caching={self.enable_caching!r}, "
            f"use_real_api={self.use_real_api!r}, ttl_seconds={self.ttl_seconds!r}, "
            f"origin={dict(self.origin)!r})"
        )

    def __repr__(self) -> str:
        """Repr with redacted API key for safe debugging."""
        return self.__str__()

    def to_frozen(self) -> "FrozenConfig":
        """Convert to the immutable configuration used in the pipeline.

        Returns:
            FrozenConfig with the same field values, excluding audit metadata.
        """
        return FrozenConfig(
            api_key=self.api_key,
            model=self.model,
            tier=self.tier,
            enable_caching=self.enable_caching,
            use_real_api=self.use_real_api,
            ttl_seconds=self.ttl_seconds,
        )

    def with_overrides(self, **overrides: object) -> "ResolvedConfig":
        """Create a new ResolvedConfig with programmatic overrides applied.

        This is useful for scoped configuration changes or test setup.

        Args:
            **overrides: Field values to override. Unknown fields are ignored.

        Returns:
            New ResolvedConfig with overrides applied and origin updated.
        """
        # Start with current values and origins
        new_values = self._asdict()
        new_origin = dict(self.origin)

        # Apply overrides and mark their origin
        for field, value in overrides.items():
            if hasattr(self, field):  # Only override known fields
                new_values[field] = value
                new_origin[field] = "programmatic"

        # Update origin in the new values dict
        new_values["origin"] = new_origin

        return ResolvedConfig(**new_values)

    def audit(self) -> str:
        """Generate a redacted audit report showing the origin of each field.

        This shows where each configuration value came from without revealing
        sensitive information like API keys.

        Returns:
            Human-readable audit report with redacted sensitive fields.
        """
        lines = []

        # Field display order for consistent output
        field_order = [
            "api_key",
            "model",
            "tier",
            "enable_caching",
            "use_real_api",
            "ttl_seconds",
        ]

        for field in field_order:
            if field in self.origin:
                origin = self.origin[field]

                # Redact sensitive fields
                if field == "api_key":
                    actual_value = getattr(self, field)
                    if actual_value is None:
                        value_display = f"{origin}:None"
                    elif origin == "env":
                        value_display = "env:[REDACTED]"
                    elif origin == "file":
                        value_display = "file:<redacted>"
                    elif origin == "programmatic":
                        value_display = "programmatic:<redacted>"
                    else:
                        value_display = f"{origin}:<redacted>"
                else:
                    # Non-sensitive fields show actual values
                    actual_value = getattr(self, field)
                    if origin == "env":
                        value_display = f"env:GEMINI_{field.upper()}={actual_value}"
                    elif origin == "file":
                        value_display = f"file:{actual_value}"
                    else:
                        value_display = f"{origin}:{actual_value}"

                lines.append(f"{field}: {value_display}")

        return "\n".join(lines)


@dataclass(frozen=True)
class FrozenConfig:
    """Immutable configuration attached to commands in the pipeline.

    This is the final form of configuration that flows through the pipeline.
    It contains only the essential field values without audit metadata,
    and is guaranteed to be immutable.

    Pipeline handlers receive this object and access fields as attributes.
    Any attempt to modify this object will raise an exception.
    """

    api_key: str | None
    model: str
    tier: APITier
    enable_caching: bool
    use_real_api: bool
    ttl_seconds: int

    def __str__(self) -> str:
        """String representation with redacted API key for safe logging."""
        api_key_display = "[REDACTED]" if self.api_key else None
        return (
            f"FrozenConfig(api_key={api_key_display!r}, model={self.model!r}, "
            f"tier={self.tier!r}, enable_caching={self.enable_caching!r}, "
            f"use_real_api={self.use_real_api!r}, ttl_seconds={self.ttl_seconds!r})"
        )

    def __repr__(self) -> str:
        """Representation with redacted API key for safe debugging."""
        return self.__str__()
