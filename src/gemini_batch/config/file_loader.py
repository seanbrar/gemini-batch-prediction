"""File-based configuration loading with profile support.

This module handles loading configuration from TOML files, supporting both
project-level (pyproject.toml) and home-level (~/.config/gemini_batch.toml)
configuration with named profiles.
"""

from pathlib import Path
import sys
from typing import Any

# Use stdlib tomllib on Python 3.11+, fallback to tomli
# Minimum Python version is 3.13, but this allows flexibility for lower versions later
if sys.version_info >= (3, 11):  # noqa: UP036
    import tomllib
else:
    import tomli as tomllib


class ConfigFileError(Exception):
    """Raised when configuration file loading fails."""

    def __init__(
        self, file_path: Path, message: str, cause: Exception | None = None
    ) -> None:
        """Initialize with file path, message, and optional cause.

        Args:
            file_path: The file that failed to load
            message: Human-readable error message
            cause: The underlying exception that caused the failure
        """
        self.file_path = file_path
        self.message = message
        self.cause = cause
        super().__init__(f"Config file error in {file_path}: {message}")


class FileConfigLoader:
    """Loads configuration from TOML files with profile support.

    This class handles loading from both project-level pyproject.toml files
    and home-level configuration files, with support for named profiles.
    """

    def __init__(self) -> None:
        """Initialize the file config loader."""

    def load_project_config(
        self, project_root: Path | None = None, profile: str | None = None
    ) -> dict[str, Any]:
        """Load configuration from pyproject.toml in the project root.

        Args:
            project_root: Directory to search for pyproject.toml. If None,
                         searches current directory and parents.
            profile: Optional profile name to load from [tool.gemini_batch.profiles.<name>]
                    If None, loads from [tool.gemini_batch]

        Returns:
            Dictionary of configuration values from the file.
            Empty dict if file doesn't exist or has no gemini_batch section.

        Raises:
            ConfigFileError: If file exists but cannot be parsed or has invalid format.
        """
        pyproject_path = self._find_pyproject_toml(project_root)
        if not pyproject_path:
            return {}

        try:
            with Path(pyproject_path).open(mode="rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise ConfigFileError(
                pyproject_path, f"Failed to parse TOML: {e}", cause=e
            ) from e

        # Navigate to the gemini_batch configuration
        tool_config = data.get("tool", {})
        gemini_config = tool_config.get("gemini_batch", {})

        if not gemini_config:
            return {}

        # Handle profile selection
        if profile:
            profiles = gemini_config.get("profiles", {})
            if profile not in profiles:
                available = list(profiles.keys()) if profiles else []
                raise ConfigFileError(
                    pyproject_path,
                    f"Profile '{profile}' not found. Available profiles: {available}",
                )
            return dict(profiles[profile])
        # Return base configuration, excluding profiles section
        config = dict(gemini_config)
        config.pop("profiles", None)  # Remove profiles section from base config
        return config

    def load_home_config(self, profile: str | None = None) -> dict[str, Any]:
        """Load configuration from ~/.config/gemini_batch.toml.

        Args:
            profile: Optional profile name to load from [profiles.<name>]
                    If None, loads from the root level.

        Returns:
            Dictionary of configuration values from the home file.
            Empty dict if file doesn't exist.

        Raises:
            ConfigFileError: If file exists but cannot be parsed or has invalid format.
        """
        home_config_path = self._get_home_config_path()
        if not home_config_path.exists():
            return {}

        try:
            with Path(home_config_path).open(mode="rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise ConfigFileError(
                home_config_path, f"Failed to parse TOML: {e}", cause=e
            ) from e

        # Handle profile selection
        if profile:
            profiles = data.get("profiles", {})
            if profile not in profiles:
                available = list(profiles.keys()) if profiles else []
                raise ConfigFileError(
                    home_config_path,
                    f"Profile '{profile}' not found. Available profiles: {available}",
                )
            return dict(profiles[profile])
        # Return base configuration, excluding profiles section
        config = dict(data)
        config.pop("profiles", None)  # Remove profiles section from base config
        return config

    def list_available_profiles(
        self, project_root: Path | None = None
    ) -> dict[str, list[str]]:
        """List all available profiles from project and home files.

        Args:
            project_root: Directory to search for pyproject.toml

        Returns:
            Dictionary with 'project' and 'home' keys containing lists of profile names.
        """
        profiles: dict[str, list[str]] = {"project": [], "home": []}

        # Check project profiles
        try:
            pyproject_path = self._find_pyproject_toml(project_root)
            if pyproject_path:
                with Path(pyproject_path).open(mode="rb") as f:
                    data = tomllib.load(f)
                tool_config = data.get("tool", {})
                gemini_config = tool_config.get("gemini_batch", {})
                project_profiles = gemini_config.get("profiles", {})
                profiles["project"] = list(project_profiles.keys())
        except (OSError, tomllib.TOMLDecodeError):
            # Ignore parsing errors for profile listing
            pass

        # Check home profiles
        try:
            home_config_path = self._get_home_config_path()
            if home_config_path.exists():
                with Path(home_config_path).open(mode="rb") as f:
                    data = tomllib.load(f)
                home_profiles = data.get("profiles", {})
                profiles["home"] = list(home_profiles.keys())
        except (OSError, tomllib.TOMLDecodeError):
            # Ignore parsing errors for profile listing
            pass

        return profiles

    def _find_pyproject_toml(self, start_dir: Path | None = None) -> Path | None:
        """Find pyproject.toml by searching up the directory tree.

        Args:
            start_dir: Directory to start searching from. If None, uses current directory.

        Returns:
            Path to pyproject.toml if found, None otherwise.
        """
        if start_dir is None:
            start_dir = Path.cwd()

        current = Path(start_dir).resolve()

        # Search up the directory tree
        while current != current.parent:  # Stop at filesystem root
            pyproject_path = current / "pyproject.toml"
            if pyproject_path.exists():
                return pyproject_path
            current = current.parent

        return None

    def _get_home_config_path(self) -> Path:
        """Get the path to the home configuration file.

        Returns:
            Path to ~/.config/gemini_batch.toml
        """
        return Path.home() / ".config" / "gemini_batch.toml"
