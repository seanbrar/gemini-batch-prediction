"""Configuration schema and validation using Pydantic.

This module defines the settings schema that validates and coerces configuration
values from various sources (environment, files, programmatic) into the correct
types with proper defaults.
"""

from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from gemini_batch.core.models import APITier


class GeminiSettings(BaseSettings):
    """Pydantic settings schema for Gemini configuration.

    This handles validation, type coercion, and default values for all
    configuration fields. It integrates with environment variables using
    the GEMINI_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        env_file=None,  # Only used when explicitly requested
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore unknown env vars for forward compatibility
    )

    # --- Core Configuration Fields ---

    api_key: str | None = Field(
        default=None,
        description="Google Gemini API key",
        # Note: validation that api_key is required when use_real_api=True
        # is handled by the model_validator below
    )

    model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model identifier",
        min_length=1,
    )

    tier: APITier = Field(
        default=APITier.FREE,
        description="API billing tier for rate limiting",
    )

    enable_caching: bool = Field(
        default=True,
        description="Whether to enable context caching",
    )

    use_real_api: bool = Field(
        default=False,
        description="Use real API instead of mock (library setting)",
    )

    ttl_seconds: int = Field(
        default=3600,
        description="TTL for cached content in seconds",
        ge=1,  # Must be at least 1 second
    )

    # --- Validation Rules ---

    @field_validator("tier", mode="before")
    @classmethod
    def parse_tier(cls, v: Any) -> APITier:
        """Parse API tier from string or enum value."""
        if isinstance(v, APITier):
            return v
        if isinstance(v, str):
            # Support both enum names and values
            tier_map = {
                "free": APITier.FREE,
                "tier_1": APITier.TIER_1,
                "tier_2": APITier.TIER_2,
                "tier_3": APITier.TIER_3,
                # Also support the enum values directly
                "FREE": APITier.FREE,
                "TIER_1": APITier.TIER_1,
                "TIER_2": APITier.TIER_2,
                "TIER_3": APITier.TIER_3,
            }
            normalized = v.lower()
            if normalized in tier_map:
                return tier_map[normalized]

        raise ValueError(
            f"Invalid tier: {v}. Must be one of: free, tier_1, tier_2, tier_3"
        )

    @model_validator(mode="after")
    def validate_api_key_requirement(self) -> "GeminiSettings":
        """Ensure api_key is provided when use_real_api is True."""
        if self.use_real_api and not self.api_key:
            raise ValueError(
                "api_key is required when use_real_api=True. "
                "Set GEMINI_API_KEY environment variable, provide in config file, "
                "or pass programmatically."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary suitable for SourceMap annotation.

        Returns:
            Dictionary with field names as keys and resolved values.
        """
        return {
            "api_key": self.api_key,
            "model": self.model,
            "tier": self.tier,
            "enable_caching": self.enable_caching,
            "use_real_api": self.use_real_api,
            "ttl_seconds": self.ttl_seconds,
        }
