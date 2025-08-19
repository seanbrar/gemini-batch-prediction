"""Unit tests for the new configuration system.

These tests verify the core behaviors of the configuration module:
- Correctly loading settings from environment variables.
- Prioritizing explicit configuration over ambient context.
- Ensuring the `config_scope` context manager works as expected.
"""

import os
from unittest.mock import patch

import pytest

from gemini_batch.config import config_scope, resolve_config
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
            resolved = resolve_config()

            # Assert
            assert resolved.api_key == "env-api-key-123"
            assert resolved.model == "env-model-flash"

    @pytest.mark.unit
    def test_get_ambient_config_uses_defaults(self):
        """
        Verifies that defaults are used for optional settings when not provided.
        """
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-api-key-only"}):
            # Act
            resolved = resolve_config()

            # Assert
            assert resolved.api_key == "env-api-key-only"
            assert resolved.model == "gemini-2.0-flash"  # Default value

    @pytest.mark.unit
    def test_get_ambient_config_raises_error_if_key_is_missing(self):
        """
        Verifies that when no API key is present, the resolver returns a ResolvedConfig
        with api_key set to None (the new system does not raise for missing keys by default).
        """
        # Arrange: Ensure the key is not in the environment.
        with patch.dict(os.environ, {}, clear=True):
            # Act
            resolved = resolve_config()

            # Assert: new behavior — no exception, but api_key is None
            assert resolved.api_key is None

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
            # Create an explicit ResolvedConfig for the scope using programmatic overrides
            scoped_resolved = resolve_config(
                programmatic={
                    "api_key": "scoped-api-key-override",
                    "model": "scoped-model-override",
                }
            )

            # Act: Enter the new config_scope using a ResolvedConfig
            with config_scope(scoped_resolved):
                active = resolve_config()
                assert active.api_key == "scoped-api-key-override"
                assert active.model == "scoped-model-override"

            # After exiting the scope, resolution should reflect the environment again
            reverted = resolve_config()
            assert reverted.api_key == "env-api-key-original"
            assert reverted.model == "env-model-original"

    @pytest.mark.unit
    def test_create_executor_uses_ambient_config(self):
        """
        Verifies that the `create_executor` factory uses ambient config by default.
        """
        with patch.dict(os.environ, {"GEMINI_API_KEY": "ambient-for-executor"}):
            # Act
            executor = create_executor()

            # Assert: Executor provides FrozenConfig
            assert executor.config.api_key == "ambient-for-executor"
            assert executor.config.model == "gemini-2.0-flash"

    @pytest.mark.unit
    def test_create_executor_prioritizes_explicit_config(self):
        """
        Verifies that `create_executor` uses the explicitly passed config
        object instead of the ambient one.
        """
        # Arrange: Set an ambient config that should be ignored.
        with patch.dict(os.environ, {"GEMINI_API_KEY": "should-be-ignored"}):
            # Build an explicit FrozenConfig via resolver
            explicit_resolved = resolve_config(
                programmatic={
                    "api_key": "explicit-api-key-winner",
                    "model": "explicit-model-winner",
                }
            )
            executor = create_executor(config=explicit_resolved.to_frozen())

            # Assert: Executor provides FrozenConfig
            assert executor.config.api_key == "explicit-api-key-winner"
            assert executor.config.model == "explicit-model-winner"

    @pytest.mark.unit
    def test_use_real_api_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError):
            resolve_config(programmatic={"use_real_api": True})
