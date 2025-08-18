"""Configuration resolution with precedence handling.

This module implements the core resolution algorithm that merges configuration
from multiple sources according to the documented precedence order:
Programmatic > Environment > Project file > Home file > Defaults
"""

import os
from pathlib import Path
from typing import Any

from .audit import SourceTracker
from .env_loader import EnvironmentConfigLoader
from .file_loader import ConfigFileError, FileConfigLoader
from .schema import GeminiSettings
from .types import ResolvedConfig


class ConfigResolver:
    """Resolves configuration from multiple sources with proper precedence.

    This class implements the core resolution algorithm that merges configuration
    values from all sources according to the specified precedence order.
    """

    def __init__(self) -> None:
        """Initialize the configuration resolver."""
        self.file_loader = FileConfigLoader()
        self.env_loader = EnvironmentConfigLoader()

    def resolve(
        self,
        programmatic: dict[str, Any] | None = None,
        *,
        profile: str | None = None,
        use_env_file: str | Path | None = None,
        project_root: Path | None = None,
    ) -> ResolvedConfig:
        """Resolve configuration from all sources with proper precedence.

        Args:
            programmatic: Programmatic overrides (highest precedence)
            profile: Profile name to load from files
            use_env_file: Optional .env file to load
            project_root: Directory to search for pyproject.toml

        Returns:
            ResolvedConfig with merged values and source tracking.

        Raises:
            ValueError: If validation fails or required values are missing.
            ConfigFileError: If configuration files are malformed.
        """
        # Initialize source tracking and configuration accumulator
        source_tracker = SourceTracker()
        merged_config: dict[str, Any] = {}

        # Get profile from environment if not explicitly provided
        if profile is None:
            profile = os.getenv("GEMINI_PROFILE")

        # Step 1: Start with schema defaults
        default_settings = GeminiSettings()
        defaults = default_settings.to_dict()

        for field, value in defaults.items():
            merged_config[field] = value
            source_tracker.set_origin(field, "default")

        # Step 2: Apply home file configuration (lower precedence)
        try:
            home_config = self.file_loader.load_home_config(profile=profile)
            for field, value in home_config.items():
                if field in merged_config:  # Only override known fields
                    merged_config[field] = value
                    source_tracker.set_origin(field, "file")
        except ConfigFileError:
            # Home config errors are non-fatal - just skip home config
            pass

        # Step 3: Apply project file configuration
        try:
            project_config = self.file_loader.load_project_config(
                project_root=project_root, profile=profile
            )
            for field, value in project_config.items():
                if field in merged_config:  # Only override known fields
                    merged_config[field] = value
                    source_tracker.set_origin(field, "file")
        except ConfigFileError:
            # Project config errors are non-fatal for optional profiles
            # but should be raised for invalid files
            if profile is None:
                # Base configuration errors should be raised
                raise
            # Profile-specific errors can be skipped

        # Step 4: Apply environment variables
        try:
            env_config = self.env_loader.load_env_config(env_file=use_env_file)
            for field, value in env_config.items():
                if field in merged_config:  # Only override known fields
                    merged_config[field] = value
                    source_tracker.set_origin(field, "env")
        except (ValueError, FileNotFoundError) as e:
            # Environment errors should be raised as they indicate misconfiguration
            raise ValueError(f"Environment configuration error: {e}") from e

        # Step 5: Apply programmatic overrides (highest precedence)
        if programmatic:
            for field, value in programmatic.items():
                if field in merged_config:  # Only override known fields
                    merged_config[field] = value
                    source_tracker.set_origin(field, "programmatic")

        # Step 6: Validate the final configuration using Pydantic
        try:
            validated_settings = GeminiSettings(**merged_config)
            final_config = validated_settings.to_dict()
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}") from e

        # Step 7: Create ResolvedConfig with source tracking
        return ResolvedConfig(
            api_key=final_config["api_key"],
            model=final_config["model"],
            tier=final_config["tier"],
            enable_caching=final_config["enable_caching"],
            use_real_api=final_config["use_real_api"],
            ttl_seconds=final_config["ttl_seconds"],
            origin=source_tracker.get_source_map(),
        )

    def validate_profile_exists(
        self, profile: str, project_root: Path | None = None
    ) -> tuple[bool, bool]:
        """Check if a profile exists in project or home configuration.

        Args:
            profile: The profile name to check
            project_root: Directory to search for pyproject.toml

        Returns:
            Tuple of (exists_in_project, exists_in_home)
        """
        available_profiles = self.file_loader.list_available_profiles(project_root)

        exists_in_project = profile in available_profiles["project"]
        exists_in_home = profile in available_profiles["home"]

        return exists_in_project, exists_in_home

    def get_effective_profile(self) -> str | None:
        """Get the effective profile name from environment or None.

        Returns:
            Profile name from GEMINI_PROFILE environment variable, or None.
        """
        return os.getenv("GEMINI_PROFILE")

    def list_available_profiles(
        self, project_root: Path | None = None
    ) -> dict[str, list[str]]:
        """List all available profiles from project and home files.

        Args:
            project_root: Directory to search for pyproject.toml

        Returns:
            Dictionary with 'project' and 'home' keys containing lists of profile names.
        """
        return self.file_loader.list_available_profiles(project_root)
