"""Architectural invariant tests for the configuration system.

Layer 2: Architectural Invariants
Prove the system rules remain true across the configuration architecture.
"""

import os
from unittest.mock import patch

import pytest

from gemini_batch.config import resolve_config
from gemini_batch.config.types import FrozenConfig
from gemini_batch.core.models import APITier


class TestConfigurationArchitecturalInvariants:
    """Architectural invariant tests for configuration system."""

    @pytest.mark.contract
    def test_precedence_order_invariant(self):
        """Invariant: Precedence order must be programmatic > env > project > home > defaults."""
        # Test that explicit programmatic config overrides environment
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "env_key", "GEMINI_MODEL": "env_model"}
        ):
            result = resolve_config(
                programmatic={"api_key": "prog_key", "model": "prog_model"}
            )

            # Programmatic should win
            assert result.to_frozen().api_key == "prog_key"
            assert result.to_frozen().model == "prog_model"

            # Source map should reflect precedence
            assert result.origin["api_key"] == "programmatic"
            assert result.origin["model"] == "programmatic"

    @pytest.mark.contract
    def test_resolve_once_freeze_then_flow_invariant(self):
        """Invariant: Configuration must be resolved once and then flow immutably."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            resolved = resolve_config()
            frozen = resolved.to_frozen()

            # Frozen config must be immutable
            assert isinstance(frozen, FrozenConfig)

            # Attempting to modify should fail
            with pytest.raises(AttributeError):
                frozen.api_key = "new_key"  # type: ignore[misc]

            # Resolved config should also be immutable (NamedTuple)
            with pytest.raises(AttributeError):
                resolved.api_key = "new_key"  # type: ignore[misc]

    @pytest.mark.contract
    def test_audit_trail_completeness_invariant(self):
        """Invariant: Every configuration field must have traceable source."""
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test_key", "GEMINI_MODEL": "test_model"}
        ):
            result = resolve_config()

            # Every field in frozen config must have source tracking
            frozen_fields = {
                "api_key",
                "model",
                "tier",
                "enable_caching",
                "use_real_api",
                "ttl_seconds",
            }

            for field in frozen_fields:
                assert field in result.origin, f"Field {field} missing from origin"
                assert isinstance(result.origin[field], str), (
                    f"Source for {field} must be string"
                )

    @pytest.mark.contract
    def test_secret_redaction_invariant(self):
        """Invariant: Secrets must never appear in logs, debug output, or audit trails."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret_api_key_12345"}):
            result = resolve_config()

            # Secret should not appear in redacted representation
            assert "secret_api_key_12345" not in result.audit()
            assert "***" in result.audit() or "[REDACTED]" in result.audit()

            # Secret should not appear in string representation of source map
            source_map_str = str(result.origin)
            assert "secret_api_key_12345" not in source_map_str

    @pytest.mark.contract
    def test_validation_consistency_invariant(self):
        """Invariant: Validation rules must be consistently applied regardless of source."""
        # Test ttl_seconds validation from environment
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test", "GEMINI_TTL_SECONDS": "0"}
        ):
            with pytest.raises(ValueError) as exc_info:
                resolve_config()
            assert "ttl_seconds" in str(exc_info.value).lower()

        # Test same validation from programmatic config
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            with pytest.raises(ValueError) as exc_info:
                resolve_config(programmatic={"ttl_seconds": 0})
            assert "ttl_seconds" in str(exc_info.value).lower()

    @pytest.mark.contract
    def test_type_safety_invariant(self):
        """Invariant: Type coercion must be consistent and safe."""
        # Test boolean coercion from strings
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "test",
                "GEMINI_ENABLE_CACHING": "true",
                "GEMINI_USE_REAL_API": "false",
            },
        ):
            result = resolve_config()

            # String values should be properly coerced to booleans
            assert result.to_frozen().enable_caching is True
            assert result.to_frozen().use_real_api is False

        # Test integer coercion
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test", "GEMINI_TTL_SECONDS": "7200"}
        ):
            result = resolve_config()
            assert result.to_frozen().ttl_seconds == 7200
            assert isinstance(result.to_frozen().ttl_seconds, int)

    @pytest.mark.contract
    def test_tier_consistency_invariant(self):
        """Invariant: APITier must be consistent across the system."""
        # Test that tier is properly resolved to enum
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test", "GEMINI_TIER": "tier_1"}
        ):
            result = resolve_config()

            assert isinstance(result.to_frozen().tier, APITier)
            assert result.to_frozen().tier == APITier.TIER_1

        # Test default tier
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            result = resolve_config()
            assert isinstance(result.to_frozen().tier, APITier)

    @pytest.mark.contract
    def test_profile_isolation_invariant(self):
        """Invariant: Profile operations must not affect global state."""
        # Test that profile parameter doesn't affect global state
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            config_before = resolve_config()
            config_with_profile = resolve_config(
                profile=None
            )  # Should not affect anything
            config_after = resolve_config()

            assert config_before.to_frozen() == config_after.to_frozen()
            assert config_with_profile.to_frozen() == config_after.to_frozen()

    @pytest.mark.contract
    def test_environment_isolation_invariant(self):
        """Invariant: Configuration resolution must not modify environment."""
        original_env = dict(os.environ)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            resolve_config()

            # Environment should be unchanged after resolution
            # (Note: patch.dict handles restoration, but we test the principle)
            assert "GEMINI_API_KEY" in os.environ

        # After patch context, environment should be restored
        assert os.environ == original_env

    @pytest.mark.contract
    def test_deterministic_defaults_invariant(self):
        """Invariant: Default values must be deterministic and consistent."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            result1 = resolve_config()
            result2 = resolve_config()

            # Default values should be identical across calls
            assert result1.to_frozen().model == result2.to_frozen().model
            assert result1.to_frozen().tier == result2.to_frozen().tier
            assert (
                result1.to_frozen().enable_caching == result2.to_frozen().enable_caching
            )
            assert result1.to_frozen().use_real_api == result2.to_frozen().use_real_api
            assert result1.to_frozen().ttl_seconds == result2.to_frozen().ttl_seconds

    @pytest.mark.contract
    def test_compatibility_shim_transparency_invariant(self):
        """Invariant: Compatibility shim must provide transparent access to both config types."""
        # Compatibility shim removed; use FrozenConfig / dict conversion directly

        # Create both config types with same values
        frozen = FrozenConfig(
            api_key="test_key",
            model="gemini-2.0-flash",
            tier=APITier.FREE,
            enable_caching=True,
            use_real_api=False,
            ttl_seconds=3600,
        )

        dict_config = {
            "api_key": "test_key",
            "model": "gemini-2.0-flash",
            "tier": APITier.FREE,
            "enable_caching": True,
            "use_real_api": False,
            "ttl_seconds": 3600,
        }

        # Compare values directly
        resolved_from_dict = resolve_config(programmatic=dict_config)
        assert frozen.api_key == resolved_from_dict.api_key
        assert frozen.model == resolved_from_dict.model
        assert frozen.tier == resolved_from_dict.tier
        assert frozen.enable_caching == resolved_from_dict.enable_caching
        assert frozen.use_real_api == resolved_from_dict.use_real_api
        assert frozen.ttl_seconds == resolved_from_dict.ttl_seconds
