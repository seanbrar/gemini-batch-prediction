"""Integration behavior tests for configuration with pipeline flow.

Layer 3: Integration Behavior
Prove the pipeline exhibits expected emergent behavior with the new configuration system.
"""

import os
from unittest.mock import patch

import pytest

from gemini_batch.config import resolve_config
from gemini_batch.config.compatibility import ConfigCompatibilityShim
from gemini_batch.config.types import FrozenConfig
from gemini_batch.core.types import InitialCommand
from gemini_batch.executor import GeminiExecutor, create_executor
from gemini_batch.extensions.conversation import ConversationManager


class TestConfigurationIntegrationBehavior:
    """Integration behavior tests for configuration system with pipeline flow."""

    @pytest.mark.workflows
    def test_executor_creation_with_auto_resolution(self):
        """Integration: create_executor() should use new configuration system."""
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test_key", "GEMINI_MODEL": "test_model"}
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
            executor = create_executor(config=explicit_config)

            # Should use explicit config (converted to FrozenConfig)
            assert isinstance(executor.config, FrozenConfig)
            assert executor.config.api_key == "explicit_key"
            assert executor.config.model == "explicit_model"

    @pytest.mark.workflows
    def test_initial_command_creation_with_frozen_config(self):
        """Integration: InitialCommand should accept FrozenConfig."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            resolved = resolve_config()

            command = InitialCommand(
                sources=("test.txt",),
                prompts=("Test prompt",),
                config=resolved.frozen,
                history=(),
            )

            # Command should contain FrozenConfig
            assert isinstance(command.config, FrozenConfig)
            assert command.config.api_key == "test_key"

    @pytest.mark.workflows
    def test_conversation_manager_integration(self):
        """Integration: ConversationManager should work with auto-resolution executor."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            executor = create_executor()
            manager = ConversationManager(["test.txt"], executor)

            # Should create successfully with FrozenConfig
            assert manager is not None
            assert isinstance(manager.executor.config, FrozenConfig)

    @pytest.mark.workflows
    def test_compatibility_shim_in_pipeline_flow(self):
        """Integration: Compatibility shim should work throughout pipeline."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            executor = create_executor()

            # Create shim from executor config
            shim = ConfigCompatibilityShim(executor.config)

            # Should provide transparent access
            assert shim.api_key == "test_key"
            assert shim.model == executor.config.model
            assert shim.tier == executor.config.tier

    @pytest.mark.workflows
    def test_config_precedence_in_executor_flow(self):
        """Integration: Configuration precedence should work through executor creation."""
        # Set environment baseline
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "env_key", "GEMINI_MODEL": "env_model"}
        ):
            # Override with explicit config
            explicit_config = {"model": "explicit_model"}
            executor = create_executor(config=explicit_config)

            # Should combine correctly (explicit overrides env)
            assert executor.config.api_key == "env_key"  # From env
            assert executor.config.model == "explicit_model"  # From explicit

    @pytest.mark.workflows
    def test_configuration_error_propagation(self):
        """Integration: Configuration errors should propagate cleanly through executor."""
        # Missing API key should fail gracefully
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                create_executor()

            # Error should be descriptive
            assert "api_key" in str(exc_info.value).lower()

    @pytest.mark.workflows
    def test_dual_config_type_support_during_migration(self):
        """Integration: System should handle both FrozenConfig and dict during migration."""
        # Test with dict config (legacy)
        dict_config = {"api_key": "dict_key", "model": "dict_model"}

        executor_dict = GeminiExecutor(config=dict_config)

        # Test with FrozenConfig (new)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "frozen_key"}):
            resolved = resolve_config()
            executor_frozen = GeminiExecutor(config=resolved.frozen)

        # Both should work
        shim_dict = ConfigCompatibilityShim(executor_dict.config)
        shim_frozen = ConfigCompatibilityShim(executor_frozen.config)

        assert shim_dict.api_key == "dict_key"
        assert shim_frozen.api_key == "frozen_key"

    @pytest.mark.workflows
    def test_telemetry_integration_with_configuration(self):
        """Integration: Telemetry should work with configuration resolution."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            # Resolution should not fail with telemetry
            resolved = resolve_config()

            # Should have telemetry-safe representation
            assert resolved.redacted_repr is not None
            assert "test_key" not in resolved.redacted_repr

    @pytest.mark.workflows
    def test_config_validation_in_executor_pipeline(self):
        """Integration: Config validation should work in executor creation."""
        # Test invalid ttl_seconds
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test"}),
            pytest.raises(ValueError),
        ):
            create_executor(config={"ttl_seconds": 0})

    @pytest.mark.workflows
    def test_environment_override_behavior(self):
        """Integration: Environment variables should properly override in pipeline."""
        base_env = {"GEMINI_API_KEY": "base_key", "GEMINI_MODEL": "base_model"}

        with patch.dict(os.environ, base_env):
            executor1 = create_executor()

            # Override one variable
            with patch.dict(os.environ, {"GEMINI_MODEL": "override_model"}):
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

            # Create initial command
            initial = InitialCommand(
                sources=("test.txt",),
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
