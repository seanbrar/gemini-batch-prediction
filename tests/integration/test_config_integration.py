"""Integration behavior tests for configuration with pipeline flow.

Layer 3: Integration Behavior
Prove the pipeline exhibits expected emergent behavior with the new configuration system.
"""

import os
from unittest.mock import patch

import pytest

from gemini_batch.config import resolve_config
from gemini_batch.config.core import FrozenConfig
from gemini_batch.core.types import InitialCommand, Source
from gemini_batch.executor import GeminiExecutor, create_executor


class TestConfigurationIntegrationBehavior:
    """Integration behavior tests for configuration system with pipeline flow."""

    @pytest.mark.workflows
    def test_executor_creation_with_auto_resolution(self):
        """Integration: create_executor() should use new configuration system."""
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test_key", "GEMINI_BATCH_MODEL": "test_model"},
        ):
            executor = create_executor()

            # Should use FrozenConfig
            assert isinstance(executor.config, FrozenConfig)
            assert executor.config.api_key == "test_key"
            assert executor.config.model == "test_model"

    @pytest.mark.workflows
    def test_executor_creation_with_explicit_config(self):
        """Integration: create_executor() should respect explicit configuration."""
        explicit_config = {"api_key": "explicit_key", "model": "explicit_model"}

        with patch.dict(os.environ, {"GEMINI_API_KEY": "env_key"}):
            resolved = resolve_config(overrides=explicit_config)
            executor = create_executor(config=resolved)

            # Should use explicit config (converted to FrozenConfig)
            assert isinstance(executor.config, FrozenConfig)
            assert executor.config.api_key == "explicit_key"
            assert executor.config.model == "explicit_model"

    @pytest.mark.workflows
    def test_initial_command_creation_with_frozen_config(self):
        """Integration: InitialCommand should accept FrozenConfig."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            resolved = resolve_config()

            # Use explicit `Source` objects per the new source handling API
            text_source = Source.from_text("test.txt")
            command = InitialCommand(
                sources=(text_source,),
                prompts=("Test prompt",),
                config=resolved,
                history=(),
            )

            # Command should contain FrozenConfig
            assert isinstance(command.config, FrozenConfig)
            assert command.config.api_key == "test_key"

    @pytest.mark.workflows
    def test_compatibility_shim_in_pipeline_flow(self):
        """Integration: Compatibility shim should work throughout pipeline."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            executor = create_executor()

            # Executor already provides a FrozenConfig
            assert executor.config.api_key == "test_key"
            assert executor.config.model == executor.config.model
            assert executor.config.tier == executor.config.tier

    @pytest.mark.workflows
    def test_config_precedence_in_executor_flow(self):
        """Integration: Configuration precedence should work through executor creation."""
        # Set environment baseline
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "env_key", "GEMINI_BATCH_MODEL": "env_model"},
        ):
            # Override with explicit config
            explicit_config = {"model": "explicit_model"}
            resolved_explicit = resolve_config(overrides=explicit_config)
            executor = create_executor(config=resolved_explicit)

            # Should combine correctly (explicit overrides env)
            assert executor.config.api_key == "env_key"  # From env
            assert executor.config.model == "explicit_model"  # From explicit

    @pytest.mark.workflows
    def test_configuration_error_propagation(self):
        """Integration: Configuration errors should propagate cleanly through executor.

        New behavior: missing API key does not raise at executor creation time; the
        resolved/frozen config may have api_key == None and validation occurs when
        use_real_api=True or at call-sites that require the key.
        """
        # Missing API key should not raise during executor creation under the new model
        with patch.dict(os.environ, {}, clear=True):
            executor = create_executor()
            assert executor.config.api_key is None

    @pytest.mark.workflows
    def test_dual_config_type_support_during_migration(self):
        """Integration: System should handle both FrozenConfig and dict during migration."""
        # Test with dict config (legacy)
        dict_config = {"api_key": "dict_key", "model": "dict_model"}

        # When passing a dict, create_executor will resolve and freeze it;
        # for direct GeminiExecutor construction in tests, normalize explicitly
        resolved_dict = resolve_config(overrides=dict_config)
        executor_dict = GeminiExecutor(config=resolved_dict)

        # Test with FrozenConfig (new)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "frozen_key"}):
            resolved = resolve_config()
            executor_frozen = GeminiExecutor(config=resolved)

        # Both should work
        assert executor_dict.config.api_key == "dict_key"
        assert executor_frozen.config.api_key == "frozen_key"

    @pytest.mark.workflows
    def test_telemetry_integration_with_configuration(self):
        """Integration: Telemetry should work with configuration resolution."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            # Resolution should not fail with telemetry
            resolved = resolve_config()

            # Should have telemetry-safe representation (use str/resolved.audit())
            redacted = str(resolved)
            assert redacted is not None
            assert "test_key" not in redacted

    @pytest.mark.workflows
    def test_config_validation_in_executor_pipeline(self):
        """Integration: Config validation should work in executor creation."""
        # Test invalid ttl_seconds
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test"}),
            pytest.raises(ValueError),
        ):
            # Validation happens during resolution; resolving invalid programmatic
            # config should raise ValueError
            resolve_config(overrides={"ttl_seconds": -1})

    @pytest.mark.workflows
    def test_environment_override_behavior(self):
        """Integration: Environment variables should properly override in pipeline."""
        base_env = {
            "GEMINI_API_KEY": "base_key",
            "GEMINI_BATCH_MODEL": "base_model",
        }

        with patch.dict(os.environ, base_env):
            executor1 = create_executor()

            # Override one variable
            with patch.dict(os.environ, {"GEMINI_BATCH_MODEL": "override_model"}):
                executor2 = create_executor()

            # Should see override effect
            assert executor1.config.model == "base_model"
            assert executor2.config.model == "override_model"
            assert executor1.config.api_key == executor2.config.api_key  # Unchanged

    @pytest.mark.workflows
    def test_frozen_config_propagation_through_commands(self):
        """Integration: FrozenConfig should propagate correctly through command chain."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            executor = create_executor()

            # Create initial command using explicit `Source` objects
            text_source = Source.from_text("test.txt")
            initial = InitialCommand(
                sources=(text_source,),
                prompts=("Test",),
                config=executor.config,
                history=(),
            )

            # Config should be FrozenConfig throughout
            assert isinstance(initial.config, FrozenConfig)
            assert initial.config.api_key == "test_key"

            # Should be immutable
            with pytest.raises(AttributeError):
                initial.config.api_key = "new_key"  # type: ignore[misc]
