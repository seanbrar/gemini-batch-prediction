"""Environment variable configuration loading.

This module handles loading configuration from environment variables with
the GEMINI_ prefix, including optional .env file support and type coercion.
"""

import os
from pathlib import Path
from typing import Any

from .schema import GeminiSettings


class EnvironmentConfigLoader:
    """Loads configuration from environment variables.

    This class handles loading from GEMINI_* environment variables and
    optional .env files, with proper type coercion.
    """

    def __init__(self) -> None:
        """Initialize the environment config loader."""

    def load_env_config(self, env_file: str | Path | None = None) -> dict[str, Any]:
        """Load configuration from environment variables.

        Args:
            env_file: Optional path to .env file to load first.
                     If provided, values from this file are loaded into
                     the environment before reading GEMINI_* variables.

        Returns:
            Dictionary of configuration values found in environment.
            Only includes fields that are actually set (not defaults).

        Raises:
            ValueError: If environment variables contain invalid values.
        """
        # Load .env file if requested (this modifies os.environ)
        if env_file:
            self._load_env_file(env_file)

        # Use Pydantic Settings to parse environment variables
        # We create a settings instance but only extract the fields
        # that were actually set in the environment (not defaults)

        # First, get current environment state
        env_values = {}

        # Check for each GEMINI_ prefixed environment variable
        gemini_env_vars = {
            "GEMINI_API_KEY": "api_key",
            "GEMINI_MODEL": "model",
            "GEMINI_TIER": "tier",
            "GEMINI_ENABLE_CACHING": "enable_caching",
            "GEMINI_USE_REAL_API": "use_real_api",
            "GEMINI_TTL_SECONDS": "ttl_seconds",
        }

        for env_var, field_name in gemini_env_vars.items():
            if env_var in os.environ:
                env_values[field_name] = os.environ[env_var]

        # If no environment variables are set, return empty dict
        if not env_values:
            return {}

        # Use Pydantic to parse and validate the environment values
        try:
            # Create a settings instance with only the env values we found
            settings = GeminiSettings(**env_values)

            # Return only the fields that were actually set in environment
            result = {}
            for field_name in env_values:
                result[field_name] = getattr(settings, field_name)

            return result

        except Exception as e:
            # Re-raise with more context about which env vars caused the issue
            env_var_list = [
                f"{env_var}={os.environ[env_var]}"
                for env_var, field_name in gemini_env_vars.items()
                if field_name in env_values
            ]
            raise ValueError(
                f"Invalid environment variable values: {', '.join(env_var_list)}. "
                f"Error: {e}"
            ) from e

    def _load_env_file(self, env_file: str | Path) -> None:
        """Load environment variables from a .env file.

        Args:
            env_file: Path to the .env file to load.

        Raises:
            FileNotFoundError: If the .env file doesn't exist.
            ValueError: If the .env file has invalid format.
        """
        env_path = Path(env_file)
        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_path}")

        try:
            with Path(env_path).open(encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue

                    # Parse KEY=VALUE format
                    if "=" not in line:
                        raise ValueError(
                            f"Invalid format at line {line_num}: {line}. "
                            "Expected KEY=VALUE format."
                        )

                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or (
                        value.startswith("'") and value.endswith("'")
                    ):
                        value = value[1:-1]

                    # Set in environment (don't override existing env vars)
                    if key not in os.environ:
                        os.environ[key] = value

        except OSError as e:
            raise ValueError(f"Failed to read environment file {env_path}: {e}") from e

    def get_env_summary(self) -> dict[str, str]:
        """Get a summary of current GEMINI_* environment variables.

        Returns:
            Dictionary mapping environment variable names to their values.
            Sensitive values (like API_KEY) are redacted.
        """
        summary = {}

        gemini_env_vars = [
            "GEMINI_API_KEY",
            "GEMINI_MODEL",
            "GEMINI_TIER",
            "GEMINI_ENABLE_CACHING",
            "GEMINI_USE_REAL_API",
            "GEMINI_TTL_SECONDS",
        ]

        for env_var in gemini_env_vars:
            if env_var in os.environ:
                if "API_KEY" in env_var:
                    summary[env_var] = "<redacted>"
                else:
                    summary[env_var] = os.environ[env_var]

        return summary
