"""Public API for the configuration system.

This module provides the main entry points for configuration resolution,
including the resolve_config() function and profile management utilities.
"""

import contextvars  # noqa: F401
from pathlib import Path
from typing import Any

from .resolver import ConfigResolver
from .types import ResolvedConfig

# Global resolver instance for efficient reuse
_resolver = ConfigResolver()

# ruff: noqa: T201


def resolve_config(
    programmatic: dict[str, Any] | None = None,
    *,
    profile: str | None = None,
    use_env_file: str | Path | None = None,
    project_root: Path | None = None,
) -> ResolvedConfig:
    """Resolve configuration from all sources with proper precedence.

    This is the main entry point for configuration resolution. It merges
    configuration from multiple sources according to the documented precedence:
    Programmatic > Environment > Project file > Home file > Defaults

    Args:
        programmatic: Dictionary of programmatic overrides (highest precedence).
                     Only known configuration fields are used.
        profile: Profile name to load from configuration files. If None,
                uses GEMINI_PROFILE environment variable if set.
        use_env_file: Optional path to .env file to load before reading
                     environment variables.
        project_root: Directory to search for pyproject.toml. If None,
                     searches current directory and parents.

    Returns:
        ResolvedConfig with merged values and source tracking for audit.

    Raises:
        ValueError: If configuration validation fails, required values are missing,
                   or environment variables contain invalid values.
        ConfigFileError: If configuration files exist but are malformed.

    Example:
        # Basic usage with environment variables
        config = resolve_config()

        # With programmatic overrides
        config = resolve_config({
            "model": "gemini-2.0-pro",
            "use_real_api": True
        })

        # With profile selection
        config = resolve_config(profile="production")

        # With .env file
        config = resolve_config(use_env_file=".env.local")
    """
    # Check if we're in a config_scope with ambient configuration
    try:
        # Try to get the ambient config context variable
        # Import here to avoid circular imports
        from .scope import _ambient_resolved_config

        ambient_config = _ambient_resolved_config.get()

        # If we have ambient config, apply programmatic overrides to it
        if programmatic:
            return ambient_config.with_overrides(**programmatic)
        return ambient_config

    except (LookupError, ImportError):
        # No ambient config or import error, do normal resolution
        return _resolver.resolve(
            programmatic=programmatic,
            profile=profile,
            use_env_file=use_env_file,
            project_root=project_root,
        )


def list_available_profiles(project_root: Path | None = None) -> dict[str, list[str]]:
    """List all available configuration profiles.

    Args:
        project_root: Directory to search for pyproject.toml. If None,
                     searches current directory and parents.

    Returns:
        Dictionary with 'project' and 'home' keys containing lists of
        available profile names from each source.

    Example:
        profiles = list_available_profiles()
        print(f"Project profiles: {profiles['project']}")
        print(f"Home profiles: {profiles['home']}")
    """
    return _resolver.list_available_profiles(project_root)


def get_effective_profile() -> str | None:
    """Get the currently effective profile name.

    Returns:
        Profile name from GEMINI_PROFILE environment variable, or None
        if no profile is set.

    Example:
        profile = get_effective_profile()
        if profile:
            print(f"Using profile: {profile}")
        else:
            print("Using default configuration")
    """
    return _resolver.get_effective_profile()


def validate_profile(profile: str, project_root: Path | None = None) -> dict[str, bool]:
    """Validate that a profile exists in available configuration files.

    Args:
        profile: The profile name to validate
        project_root: Directory to search for pyproject.toml

    Returns:
        Dictionary with 'project' and 'home' keys indicating whether
        the profile exists in each source.

    Raises:
        ValueError: If the profile doesn't exist in any configuration file.

    Example:
        try:
            exists = validate_profile("production")
            if exists['project']:
                print("Profile found in project configuration")
            elif exists['home']:
                print("Profile found in home configuration")
        except ValueError as e:
            print(f"Profile validation failed: {e}")
    """
    exists_in_project, exists_in_home = _resolver.validate_profile_exists(
        profile, project_root
    )

    if not exists_in_project and not exists_in_home:
        available = list_available_profiles(project_root)
        all_profiles = available["project"] + available["home"]
        raise ValueError(
            f"Profile '{profile}' not found. Available profiles: {all_profiles}"
        )

    return {"project": exists_in_project, "home": exists_in_home}


def print_config_audit(config: ResolvedConfig) -> None:
    """Print a human-readable audit of configuration sources.

    This is a convenience function for debugging and understanding
    where configuration values are coming from.

    Args:
        config: The resolved configuration to audit

    Example:
        config = resolve_config()
        print_config_audit(config)
        # Output:
        # api_key: env:GEMINI_API_KEY
        # model: file:gemini-2.0-flash
        # tier: default:APITier.FREE
        # ...
    """
    print(config.audit())


def print_effective_config(
    programmatic: dict[str, Any] | None = None,
    *,
    profile: str | None = None,
    redacted: bool = True,
) -> None:
    """Print the effective configuration that would be used.

    This is useful for debugging configuration issues and understanding
    the final resolved values.

    Args:
        programmatic: Programmatic overrides to apply
        profile: Profile to use
        redacted: Whether to redact sensitive values (default: True)

    Example:
        print_effective_config(profile="development")
        print_effective_config({"model": "gemini-2.0-pro"}, redacted=False)
    """
    try:
        config = resolve_config(programmatic=programmatic, profile=profile)

        print("=== Effective Configuration ===")
        if redacted:
            print(config.audit())
        else:
            print(f"api_key: {config.api_key}")
            print(f"model: {config.model}")
            print(f"tier: {config.tier}")
            print(f"enable_caching: {config.enable_caching}")
            print(f"use_real_api: {config.use_real_api}")
            print(f"ttl_seconds: {config.ttl_seconds}")

    except Exception as e:
        print(f"Configuration error: {e}")


# Convenience function for environment variable checking
def check_environment() -> dict[str, str]:
    """Check current GEMINI_* environment variables.

    Returns:
        Dictionary of currently set GEMINI_* environment variables.
        Sensitive values are redacted.

    Example:
        env_vars = check_environment()
        for var, value in env_vars.items():
            print(f"{var}: {value}")
    """
    return _resolver.env_loader.get_env_summary()
