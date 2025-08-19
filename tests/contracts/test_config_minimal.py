"""Essential contract tests for the configuration system.

Minimal, focused tests for the most critical configuration contracts.
"""

import os
from unittest.mock import patch

import pytest

from src.gemini_batch.config import resolve_config
from src.gemini_batch.config.compatibility import (
    ConfigCompatibilityShim,
    ensure_frozen_config,
)
from src.gemini_batch.config.types import FrozenConfig, ResolvedConfig
from src.gemini_batch.core.models import APITier
from src.gemini_batch.executor import create_executor


class TestEssentialConfigurationContracts:
    """Essential configuration system contract tests."""

    @pytest.mark.contract
    def test_frozen_config_immutability(self):
        """Essential: FrozenConfig must be immutable."""
        config = FrozenConfig(
            api_key="test_key",
            model="gemini-2.0-flash",
            tier=APITier.FREE,
            enable_caching=True,
            use_real_api=False,
            ttl_seconds=3600,
        )

        with pytest.raises(AttributeError):
            config.api_key = "new_key"  # type: ignore[misc]

    @pytest.mark.contract
    def test_resolve_config_basic_functionality(self):
        """Essential: resolve_config() must return proper ResolvedConfig."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            result = resolve_config()

            assert isinstance(result, ResolvedConfig)
            assert result.api_key == "test_key"
            assert hasattr(result, "origin")

            frozen = result.to_frozen()
            assert isinstance(frozen, FrozenConfig)
            assert frozen.api_key == "test_key"

    @pytest.mark.contract
    def test_compatibility_shim_works(self):
        """Essential: ConfigCompatibilityShim must handle both config types."""
        frozen = FrozenConfig(
            api_key="test_key",
            model="gemini-2.0-flash",
            tier=APITier.FREE,
            enable_caching=True,
            use_real_api=False,
            ttl_seconds=3600,
        )

        shim = ConfigCompatibilityShim(frozen)
        assert shim.api_key == "test_key"
        assert shim.model == "gemini-2.0-flash"

    @pytest.mark.contract
    def test_precedence_order_basic(self):
        """Essential: Programmatic config must override environment."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env_key"}):
            result = resolve_config(programmatic={"api_key": "prog_key"})
            assert result.api_key == "prog_key"
            assert result.origin["api_key"] == "programmatic"

    @pytest.mark.contract
    def test_executor_integration_basic(self):
        """Essential: create_executor() must work with new config system."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            executor = create_executor()
            # Executor should have some config (dict or FrozenConfig)
            assert executor.config is not None

            # Should be accessible via shim
            shim = ConfigCompatibilityShim(executor.config)
            assert shim.api_key == "test_key"

    @pytest.mark.contract
    def test_configuration_resolution_basic(self):
        """Essential: Configuration resolution must work with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            # Should resolve to defaults (validation not implemented yet)
            result = resolve_config()
            assert isinstance(result, ResolvedConfig)
            assert result.api_key is None  # Default when not provided
            assert result.model is not None  # Should have default model

    @pytest.mark.contract
    def test_ensure_frozen_config_conversion(self):
        """Essential: ensure_frozen_config() must handle both types."""
        # Test with dict
        dict_config = {
            "api_key": "test_key",
            "model": "gemini-2.0-flash",
            "tier": APITier.FREE,
            "enable_caching": True,
            "use_real_api": False,
            "ttl_seconds": 3600,
        }

        result = ensure_frozen_config(dict_config)
        assert isinstance(result, FrozenConfig)
        assert result.api_key == "test_key"

    @pytest.mark.security
    def test_api_key_not_in_string_repr_basic(self):
        """Essential: API key must not appear in string representations."""
        secret_key = "sk-very-secret-api-key-12345"

        frozen = FrozenConfig(
            api_key=secret_key,
            model="gemini-2.0-flash",
            tier=APITier.FREE,
            enable_caching=True,
            use_real_api=False,
            ttl_seconds=3600,
        )

        # Secret should not appear in string representations
        assert secret_key not in str(frozen)
        assert secret_key not in repr(frozen)

    @pytest.mark.security
    def test_source_tracking_no_secrets(self):
        """Essential: Origin tracking must not contain secret values."""
        secret_key = "sk-very-secret-api-key-12345"

        with patch.dict(os.environ, {"GEMINI_API_KEY": secret_key}):
            result = resolve_config()

            # Origin should track source but not contain secret
            assert "api_key" in result.origin
            assert secret_key not in str(result.origin)

    @pytest.mark.workflows
    def test_end_to_end_configuration_flow(self):
        """Essential: Complete flow from resolution to executor to command."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            # Resolution
            resolved = resolve_config()
            assert resolved.api_key == "test_key"

            # Conversion
            frozen = resolved.to_frozen()
            assert frozen.api_key == "test_key"

            # Executor creation
            executor = create_executor()
            shim = ConfigCompatibilityShim(executor.config)
            assert shim.api_key == "test_key"
