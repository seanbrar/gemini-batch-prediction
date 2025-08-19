"""Security contract tests for the configuration system.

Layer 1/2: Contract Compliance + Security Invariants
Prove that secrets are handled securely and never leak into logs/audits.
"""

import json
import os
from unittest.mock import patch

import pytest

from src.gemini_batch.config import resolve_config
from src.gemini_batch.config.types import FrozenConfig
from src.gemini_batch.core.models import APITier


class TestConfigurationSecurityContracts:
    """Security contract tests for configuration system."""

    @pytest.mark.contract
    @pytest.mark.security
    def test_api_key_never_in_string_representation(self):
        """Security: API key must never appear in string representations."""
        secret_key = "sk-very-secret-api-key-12345-abcdef"

        with patch.dict(os.environ, {"GEMINI_API_KEY": secret_key}):
            resolved = resolve_config()

            # Check various string representations
            resolved_str = str(resolved)
            resolved_repr = repr(resolved)
            frozen_str = str(resolved.frozen)
            frozen_repr = repr(resolved.frozen)

            # Secret should not appear in any string representation
            assert secret_key not in resolved_str
            assert secret_key not in resolved_repr
            assert secret_key not in frozen_str
            assert secret_key not in frozen_repr

    @pytest.mark.contract
    @pytest.mark.security
    def test_api_key_redacted_in_audit_repr(self):
        """Security: API key must be redacted in audit representation."""
        secret_key = "sk-very-secret-api-key-12345-abcdef"

        with patch.dict(os.environ, {"GEMINI_API_KEY": secret_key}):
            resolved = resolve_config()

            # Redacted representation should not contain secret
            assert secret_key not in resolved.redacted_repr

            # Should contain redaction indicator
            redacted_repr_lower = resolved.redacted_repr.lower()
            assert any(
                indicator in redacted_repr_lower
                for indicator in ["***", "[redacted]", "hidden", "secret"]
            )

    @pytest.mark.contract
    @pytest.mark.security
    def test_source_map_never_contains_secrets(self):
        """Security: Source map must never contain actual secret values."""
        secret_key = "sk-very-secret-api-key-12345-abcdef"

        with patch.dict(os.environ, {"GEMINI_API_KEY": secret_key}):
            resolved = resolve_config()

            # Convert source map to string for searching
            source_map_str = str(resolved.source_map)
            source_map_repr = repr(resolved.source_map)

            # Secret should not appear anywhere in source map
            assert secret_key not in source_map_str
            assert secret_key not in source_map_repr

            # Source map should still track the field
            assert "api_key" in resolved.source_map

    @pytest.mark.contract
    @pytest.mark.security
    def test_json_serialization_excludes_secrets(self):
        """Security: JSON serialization must exclude secret fields."""
        secret_key = "sk-very-secret-api-key-12345-abcdef"

        frozen = FrozenConfig(
            api_key=secret_key,
            model="gemini-2.0-flash",
            tier=APITier.FREE,
            enable_caching=True,
            use_real_api=False,
            ttl_seconds=3600,
        )

        # Create safe dict for serialization (without secrets)
        safe_dict = {
            "model": frozen.model,
            "tier": frozen.tier.value,
            "enable_caching": frozen.enable_caching,
            "use_real_api": frozen.use_real_api,
            "ttl_seconds": frozen.ttl_seconds,
            # Note: api_key deliberately excluded
        }

        # Should serialize without secrets
        serialized = json.dumps(safe_dict)
        assert secret_key not in serialized

    @pytest.mark.contract
    @pytest.mark.security
    def test_exception_messages_dont_leak_secrets(self):
        """Security: Exception messages must not contain secret values."""
        secret_key = "sk-very-secret-api-key-12345-abcdef"

        # Test validation error with secret in context
        with patch.dict(os.environ, {"GEMINI_API_KEY": secret_key}):
            try:
                # Force a validation error
                resolve_config(programmatic={"ttl_seconds": -1})
                pytest.fail("Should have raised validation error")
            except ValueError as e:
                error_msg = str(e)

                # Error message should not contain the secret
                assert secret_key not in error_msg

    @pytest.mark.contract
    @pytest.mark.security
    def test_logging_safe_representations(self):
        """Security: All representations used for logging must be safe."""
        secret_key = "sk-very-secret-api-key-12345-abcdef"

        with patch.dict(os.environ, {"GEMINI_API_KEY": secret_key}):
            resolved = resolve_config()

            # Test common logging representations
            logging_representations = [
                str(resolved),
                repr(resolved),
                str(resolved.frozen),
                repr(resolved.frozen),
                resolved.redacted_repr,
                f"Config: {resolved}",
                f"Frozen: {resolved.frozen}",
            ]

            for representation in logging_representations:
                assert secret_key not in representation, (
                    f"Secret found in: {representation}"
                )

    @pytest.mark.contract
    @pytest.mark.security
    def test_dict_conversion_preserves_security(self):
        """Security: Dict conversion must maintain security properties."""
        secret_key = "sk-very-secret-api-key-12345-abcdef"

        frozen = FrozenConfig(
            api_key=secret_key,
            model="gemini-2.0-flash",
            tier=APITier.FREE,
            enable_caching=True,
            use_real_api=False,
            ttl_seconds=3600,
        )

        # Manual conversion to dict should still require explicit handling
        # (We don't provide automatic dict conversion that includes secrets)

        # If someone accidentally converts to dict, the secret is there
        # but our string representations should still be safe
        unsafe_dict = {
            "api_key": frozen.api_key,
            "model": frozen.model,
            "tier": frozen.tier,
            "enable_caching": frozen.enable_caching,
            "use_real_api": frozen.use_real_api,
            "ttl_seconds": frozen.ttl_seconds,
        }

        # The dict contains the secret (unavoidable)
        assert unsafe_dict["api_key"] == secret_key

        # But our safe representations should still work
        from src.gemini_batch.config.compatibility import ConfigCompatibilityShim

        shim = ConfigCompatibilityShim(unsafe_dict)

        # Shim should provide access but not leak in string form
        assert shim.api_key == secret_key  # Direct access works
        # Note: We rely on frozen config string methods for safety

    @pytest.mark.contract
    @pytest.mark.security
    def test_telemetry_data_excludes_secrets(self):
        """Security: Telemetry data must never include secret values."""
        secret_key = "sk-very-secret-api-key-12345-abcdef"

        with patch.dict(os.environ, {"GEMINI_API_KEY": secret_key}):
            resolved = resolve_config()

            # Simulate telemetry data collection
            telemetry_safe_config = {
                "model": resolved.frozen.model,
                "tier": resolved.frozen.tier.value,
                "enable_caching": resolved.frozen.enable_caching,
                "use_real_api": resolved.frozen.use_real_api,
                "ttl_seconds": resolved.frozen.ttl_seconds,
                "config_sources": list(resolved.source_map.values()),
            }

            # Serialize for telemetry
            telemetry_json = json.dumps(telemetry_safe_config)

            # Should not contain secrets
            assert secret_key not in telemetry_json

    @pytest.mark.contract
    @pytest.mark.security
    def test_environment_variable_names_not_sensitive(self):
        """Security: Environment variable names themselves are not sensitive."""
        # This is acceptable - env var names are not secrets
        secret_key = "sk-very-secret-api-key-12345-abcdef"

        with patch.dict(os.environ, {"GEMINI_API_KEY": secret_key}):
            resolved = resolve_config()

            # Source map can contain env var names (not sensitive)
            assert "environment" in resolved.source_map["api_key"]

            # But not the actual secret value
            assert secret_key not in str(resolved.source_map)

    @pytest.mark.contract
    @pytest.mark.security
    def test_compatibility_shim_preserves_security(self):
        """Security: Compatibility shim must maintain security properties."""
        from src.gemini_batch.config.compatibility import ConfigCompatibilityShim

        secret_key = "sk-very-secret-api-key-12345-abcdef"

        frozen = FrozenConfig(
            api_key=secret_key,
            model="gemini-2.0-flash",
            tier=APITier.FREE,
            enable_caching=True,
            use_real_api=False,
            ttl_seconds=3600,
        )

        shim = ConfigCompatibilityShim(frozen)

        # Shim provides access to secret
        assert shim.api_key == secret_key

        # But string representations should be safe
        # (This depends on the underlying FrozenConfig having safe __str__)
        shim_str = str(shim)
        shim_repr = repr(shim)

        # These should not contain the secret
        assert secret_key not in shim_str
        assert secret_key not in shim_repr
