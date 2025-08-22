# src/gemini_batch/config/loaders.py

"""Configuration loaders for environment and files.

This module provides pure data loading functions that extract configuration
values from various sources without performing validation or complex logic.
Each loader returns plain dictionaries that the core resolver can merge.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

# Python 3.13+ has tomllib in stdlib
import tomllib

from . import utils

# Note on imports:
# Import the `utils` module at top-level (module object) but avoid binding any of
# its function results at import time. For schema-informed behaviors (like type
# coercion), import `Settings` lazily inside functions to keep the module
# side-effect free and avoid import-order coupling.


def _coerce_bool(v: str) -> bool:
    """Convert string to boolean using common conventions."""
    return v.strip().lower() in {"1", "true", "yes", "on"}


# --- Constants ---

CONFIG_TOOL_NAME = "gemini_batch"

# Meta/control variables that steer resolution but aren't config fields
META_ENV_FIELDS = {"profile", "pyproject_path", "config_home", "debug_config"}

# --- Environment Loading ---


def load_env() -> Mapping[str, Any]:
    """Load configuration from environment variables.

    Behavior:
    - Reads `GEMINI_*` environment variables and performs schema-informed type
      coercion (bool/int/float) based on `Settings` model annotations.
    - Intentionally excludes meta/control variables from the returned mapping
      to avoid polluting `FrozenConfig.extra` with non-config fields:
        `GEMINI_PROFILE`, `GEMINI_PYPROJECT_PATH`, `GEMINI_CONFIG_HOME`,
        `GEMINI_DEBUG_CONFIG`.
    - `.env` loading is handled at higher-level entry points; this function is
      pure and only reads from `os.environ`.

    Returns:
        Dictionary of configuration values with appropriate types.
    """
    config: dict[str, Any] = {}

    for key, value in os.environ.items():
        if not key.startswith(utils.ENV_PREFIX):
            continue

        field_name = key[len(utils.ENV_PREFIX) :].lower()

        # Skip meta/control variables
        if field_name in META_ENV_FIELDS:
            continue

        # Derive target type from Settings schema lazily to avoid import cycles
        try:
            from .core import Settings  # local import to keep loaders import-light

            info = Settings.model_fields.get(field_name)
            target_type = info.annotation if info is not None else None
            # Only coerce for simple primitives; let Pydantic handle others
            if target_type not in {bool, int, float}:
                target_type = None
        except Exception:
            # On any issue determining type, fall back to string (no coercion)
            target_type = None
        config[field_name] = _coerce_env_value(value, target_type)

    return config


def _coerce_env_value(value: str, target_type: type | None) -> Any:
    """Coerce env string to target type when possible.

    Falls back to original string on conversion failure or unknown type.
    """
    if target_type is bool:
        return _coerce_bool(value)
    if target_type is int:
        try:
            return int(value)
        except ValueError:
            return value
    if target_type is float:
        try:
            return float(value)
        except ValueError:
            return value
    return value


# --- Profile and file loading helpers ---


def list_profiles() -> list[str]:
    """List profile names available in home and project TOML files."""
    names: set[str] = set()
    for path in (utils.get_pyproject_path(), utils.get_home_config_path()):
        data = _read_toml(path)
        profiles = data.get("tool", {}).get(CONFIG_TOOL_NAME, {}).get("profiles", {})
        names.update(name for name in profiles if isinstance(name, str) and name)
    return sorted(names)


def validate_profile(name: str) -> bool:
    """Return True if profile exists in either home or project config.

    For better error messages when validation fails, use:
        available = list_profiles()
        if not validate_profile(name):
            raise ValueError(f"Profile '{name}' not found. Available: {available}")
    """
    if not name:
        return False
    return name in set(list_profiles())


def profile_validation_error(name: str) -> str:
    """Generate a descriptive error message for invalid profile names.

    Helper for client code that needs to provide detailed profile validation errors.
    Uses actual resolved paths including any environment variable overrides.
    """
    if not name:
        return "Profile name cannot be empty"

    available = list_profiles()

    # Get actual resolved paths (including any environment overrides)
    project_path = utils.get_pyproject_path()
    home_path = utils.get_home_config_path()

    if not available:
        return (
            f"Profile '{name}' not found. No profiles are configured in "
            f"{project_path} or {home_path}"
        )

    return f"Profile '{name}' not found. Available profiles: {', '.join(available)}"


def _read_toml(path: Path) -> dict[str, Any]:
    """Safely read a TOML file, returning empty dict on any error.

    Args:
        path: Path to the TOML file to read.

    Returns:
        Parsed TOML data as dictionary, or empty dict if file missing/invalid.
    """
    if not path.exists():
        return {}

    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        # Silently handle any parsing errors - return empty dict
        return {}


def _extract_tables(data: dict[str, Any], profile: str | None) -> dict[str, Any]:
    """Extract gemini_batch configuration tables from TOML data.

    Looks for [tool.gemini_batch] section and optionally overlays a profile
    from [tool.gemini_batch.profiles.<profile_name>].

    Args:
        data: Parsed TOML data.
        profile: Optional profile name to overlay.

    Returns:
        Merged configuration dictionary from base + profile sections.
    """
    tool_section = data.get("tool", {})
    gemini_batch_section = tool_section.get(CONFIG_TOOL_NAME, {})

    # Start with base configuration (excluding profiles)
    base_config = {k: v for k, v in gemini_batch_section.items() if k != "profiles"}

    # Overlay profile if specified
    if profile:
        profiles_section = gemini_batch_section.get("profiles", {})
        profile_config = profiles_section.get(profile, {})
        base_config.update(profile_config)

    return base_config


def _load_config_file(path: Path, profile: str | None = None) -> Mapping[str, Any]:
    """Unified TOML config file loader.

    Args:
        path: Path to the TOML file to load.
        profile: Optional profile name to use.

    Returns:
        Configuration dictionary from [tool.gemini_batch] section.
    """
    data = _read_toml(path)
    effective_profile = profile or utils.get_effective_profile()
    return _extract_tables(data, effective_profile)


def load_pyproject(profile: str | None = None) -> Mapping[str, Any]:
    """Load configuration from pyproject.toml in current working directory.

    Args:
        profile: Optional profile name to use. If None, checks GEMINI_PROFILE
            environment variable.

    Returns:
        Configuration dictionary from [tool.gemini_batch] section.
    """
    return _load_config_file(utils.get_pyproject_path(), profile)


def load_home(profile: str | None = None) -> Mapping[str, Any]:
    """Load configuration from user's home directory config file.

    Args:
        profile: Optional profile name to use. If None, checks GEMINI_PROFILE
            environment variable.

    Returns:
        Configuration dictionary from user's gemini_batch.toml file.
    """
    return _load_config_file(utils.get_home_config_path(), profile)
