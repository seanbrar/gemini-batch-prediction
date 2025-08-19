"""Contract compliance tests for the configuration system.

Layer 1: Contract Compliance
Prove each configuration component meets its type/behavior contract.
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


class TestConfigurationContractCompliance:
    """Contract compliance tests for configuration system components."""

    @pytest.mark.contract
    def test_frozen_config_is_immutable(self):
        """Contract: FrozenConfig must be immutable."""
        config = FrozenConfig(
            api_key="test_key",
            model="gemini-2.0-flash",
            tier=APITier.FREE,
            enable_caching=True,
            use_real_api=False,
            ttl_seconds=3600,
        )

        # Attempt mutation should fail
        with pytest.raises(AttributeError):
            config.api_key = "new_key"  # type: ignore[misc]

        with pytest.raises(AttributeError):
            config.model = "new_model"  # type: ignore[misc]

    @pytest.mark.contract
    def test_resolved_config_is_immutable(self):
        """Contract: ResolvedConfig must be immutable (NamedTuple)."""
        config = ResolvedConfig(
            api_key="test_key",
            model="gemini-2.0-flash",
            tier=APITier.FREE,
            enable_caching=True,
            use_real_api=False,
            ttl_seconds=3600,
            source_map={},
            redacted_repr="test",
        )

        # NamedTuple fields should be immutable
        with pytest.raises(AttributeError):
            config.api_key = "new_key"  # type: ignore[misc]

    @pytest.mark.contract
    def test_resolve_config_is_pure_function(self):
        """Contract: resolve_config() is a pure function - same inputs produce same outputs."""
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test_key", "GEMINI_MODEL": "test_model"}
        ):
            # Call multiple times with same inputs
            result1 = resolve_config()
            result2 = resolve_config()

            # Should produce identical results (excluding origin which may have timestamps)
            assert result1.api_key == result2.api_key
            assert result1.model == result2.model
            assert result1.tier == result2.tier
            assert result1.enable_caching == result2.enable_caching
            assert result1.use_real_api == result2.use_real_api
            assert result1.ttl_seconds == result2.ttl_seconds

    @pytest.mark.contract
    def test_resolve_config_returns_typed_result(self):
        """Contract: resolve_config() returns properly typed ResolvedConfig."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            result = resolve_config()

            # Must return ResolvedConfig
            assert isinstance(result, ResolvedConfig)

            # Must be able to convert to FrozenConfig
            frozen = result.to_frozen()
            assert isinstance(frozen, FrozenConfig)

            # Required fields must be present
            assert hasattr(result, "api_key")
            assert hasattr(result, "model")
            assert hasattr(result, "tier")
            assert hasattr(result, "origin")

    @pytest.mark.contract
    def test_compatibility_shim_signature_contract(self):
        """Contract: ConfigCompatibilityShim must provide unified field access."""
        frozen_config = FrozenConfig(
            api_key="test_key",
            model="gemini-2.0-flash",
            tier=APITier.FREE,
            enable_caching=True,
            use_real_api=False,
            ttl_seconds=3600,
        )

        shim = ConfigCompatibilityShim(frozen_config)

        # Must provide all required fields
        assert hasattr(shim, "api_key")
        assert hasattr(shim, "model")
        assert hasattr(shim, "tier")
        assert hasattr(shim, "enable_caching")
        assert hasattr(shim, "use_real_api")
        assert hasattr(shim, "ttl_seconds")

        # Field access must work
        assert shim.api_key == "test_key"
        assert shim.model == "gemini-2.0-flash"
        assert shim.tier == APITier.FREE

    @pytest.mark.contract
    def test_ensure_frozen_config_contract(self):
        """Contract: ensure_frozen_config() must handle both config types."""
        # Test with FrozenConfig - should return unchanged
        frozen = FrozenConfig(
            api_key="test_key",
            model="gemini-2.0-flash",
            tier=APITier.FREE,
            enable_caching=True,
            use_real_api=False,
            ttl_seconds=3600,
        )

        result = ensure_frozen_config(frozen)
        assert result is frozen  # Should be identical object

        # Test with dict - should convert
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

    @pytest.mark.contract
    def test_profile_system_deterministic(self):
        """Contract: Profile system should be deterministic when implemented."""
        # For now, test that profile parameter doesn't break resolution
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            result1 = resolve_config(profile=None)
            result2 = resolve_config(profile=None)

            # Should return identical results
            assert result1 == result2

    @pytest.mark.contract
    def test_configuration_types_are_serializable(self):
        """Contract: Configuration types must be serializable for telemetry."""
        frozen = FrozenConfig(
            api_key="test_key",
            model="gemini-2.0-flash",
            tier=APITier.FREE,
            enable_caching=True,
            use_real_api=False,
            ttl_seconds=3600,
        )

        # Should be able to extract fields for serialization
        fields = {
            "model": frozen.model,
            "tier": frozen.tier.value,  # Enum should have value
            "enable_caching": frozen.enable_caching,
            "use_real_api": frozen.use_real_api,
            "ttl_seconds": frozen.ttl_seconds,
            # Note: api_key should NOT be serialized (secret)
        }

        # All fields should be JSON-serializable types
        import json

        json.dumps(fields)  # Should not raise

    @pytest.mark.contract
    def test_no_global_mutable_state(self):
        """Contract: Configuration system must not use global mutable state."""
        # Verify multiple resolve operations don't interfere
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "key1", "GEMINI_MODEL": "model1"}
        ):
            config1 = resolve_config()

        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "key2", "GEMINI_MODEL": "model2"}
        ):
            config2 = resolve_config()

        # Results should be independent
        assert config1.frozen.api_key == "key1"
        assert config1.frozen.model == "model1"
        assert config2.frozen.api_key == "key2"
        assert config2.frozen.model == "model2"

    @pytest.mark.contract
    def test_error_handling_is_explicit(self):
        """Contract: Configuration errors must be explicit, not hidden exceptions."""
        # Test missing required API key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                resolve_config()

            # Error should be descriptive and explicit
            assert "api_key" in str(exc_info.value).lower()

        # Test invalid tier
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test", "GEMINI_TIER": "invalid_tier"}
        ):
            with pytest.raises(ValueError) as exc_info:
                resolve_config()

            # Should mention tier validation
            assert "tier" in str(exc_info.value).lower()
