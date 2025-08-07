"""Unit tests for the new configuration system.

These tests verify the core behaviors of the configuration module:
- Correctly loading settings from environment variables.
- Prioritizing explicit configuration over ambient context.
- Ensuring the `config_scope` context manager works as expected.
"""

import os
from unittest.mock import patch

import pytest

from gemini_batch.config import GeminiConfig, config_scope, get_ambient_config
from gemini_batch.executor import create_executor


class TestConfigurationSystem:
    """A test suite for the configuration resolution logic."""

    @pytest.mark.unit
    def test_get_ambient_config_from_environment(self):
        """
        Verifies that `get_ambient_config` correctly reads from environment variables.
        """
        # Arrange: Use patch.dict to temporarily set environment variables.
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "env-api-key-123",
                "GEMINI_MODEL": "env-model-flash",
            },
        ):
            # Act
            config = get_ambient_config()

            # Assert
            assert config["api_key"] == "env-api-key-123"
            assert config["model"] == "env-model-flash"

    @pytest.mark.unit
    def test_get_ambient_config_uses_defaults(self):
        """
        Verifies that defaults are used for optional settings when not provided.
        """
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-api-key-only"}):
            # Act
            config = get_ambient_config()

            # Assert
            assert config["api_key"] == "env-api-key-only"
            assert config["model"] == "gemini-2.0-flash"  # Default value

    @pytest.mark.unit
    def test_get_ambient_config_raises_error_if_key_is_missing(self):
        """
        Verifies that a ValueError is raised if the required API key is not found.
        """
        # Arrange: Ensure the key is not in the environment.
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="GEMINI_API_KEY is not set"),
        ):
            # Act & Assert
            get_ambient_config()

    @pytest.mark.unit
    def test_config_scope_overrides_environment(self):
        """
        Verifies that `config_scope` correctly overrides ambient environment variables.
        """
        # Arrange: Set a base configuration in the environment.
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "env-api-key-original",
                "GEMINI_MODEL": "env-model-original",
            },
        ):
            # Define the overriding configuration.
            scoped_config = GeminiConfig(
                api_key="scoped-api-key-override", model="scoped-model-override"
            )

            # Act: Enter the config_scope.
            with config_scope(scoped_config):
                # Retrieve the config from within the scope.
                active_config = get_ambient_config()

                # Assert: The active config should match the scoped config.
                assert active_config["api_key"] == "scoped-api-key-override"
                assert active_config["model"] == "scoped-model-override"

            # Assert: After exiting the scope, the config should revert to the environment.
            reverted_config = get_ambient_config()
            assert reverted_config["api_key"] == "env-api-key-original"
            assert reverted_config["model"] == "env-model-original"

    @pytest.mark.unit
    def test_create_executor_uses_ambient_config(self):
        """
        Verifies that the `create_executor` factory uses ambient config by default.
        """
        with patch.dict(os.environ, {"GEMINI_API_KEY": "ambient-for-executor"}):
            # Act
            executor = create_executor()

            # Assert
            assert executor.config["api_key"] == "ambient-for-executor"
            assert executor.config["model"] == "gemini-2.0-flash"

    @pytest.mark.unit
    def test_create_executor_prioritizes_explicit_config(self):
        """
        Verifies that `create_executor` uses the explicitly passed config
        object instead of the ambient one.
        """
        # Arrange: Set an ambient config that should be ignored.
        with patch.dict(os.environ, {"GEMINI_API_KEY": "should-be-ignored"}):
            explicit_config = GeminiConfig(
                api_key="explicit-api-key-winner", model="explicit-model-winner"
            )

            # Act
            executor = create_executor(config=explicit_config)

            # Assert
            assert executor.config["api_key"] == "explicit-api-key-winner"
            assert executor.config["model"] == "explicit-model-winner"
