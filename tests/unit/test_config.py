from unittest.mock import patch

import pytest

from gemini_batch.config import (
    MODEL_CAPABILITIES,
    TIER_NAMES,
    TIER_RATE_LIMITS,
    APITier,
    ConfigManager,
    ModelCapabilities,
    ModelLimits,
    RateLimits,
)
from gemini_batch.exceptions import GeminiBatchError


class TestAPITierEnum:
    """Test APITier enum functionality"""

    def test_api_tier_values(self):
        """Should have expected tier values"""
        assert APITier.FREE.value == "free"
        assert APITier.TIER_1.value == "tier_1"
        assert APITier.TIER_2.value == "tier_2"
        assert APITier.TIER_3.value == "tier_3"

    def test_api_tier_comparison(self):
        """Should support comparison operations"""
        assert APITier.FREE == APITier.FREE
        assert APITier.FREE != APITier.TIER_1


class TestDataStructureIntegrity:
    """Test the integrity and consistency of configuration data structures"""

    def test_model_capabilities_completeness(self):
        """All models in TIER_RATE_LIMITS should have corresponding capabilities"""
        all_models_in_tiers = set()
        for tier_data in TIER_RATE_LIMITS.values():
            all_models_in_tiers.update(tier_data.keys())

        capabilities_models = set(MODEL_CAPABILITIES.keys())

        missing_capabilities = all_models_in_tiers - capabilities_models
        assert not missing_capabilities, (
            f"Models missing capabilities: {missing_capabilities}"
        )

        # Verify all capability models have at least one tier assignment
        orphaned_capabilities = capabilities_models - all_models_in_tiers
        assert not orphaned_capabilities, (
            f"Capabilities without tier assignments: {orphaned_capabilities}"
        )

    def test_tier_names_completeness(self):
        """All API tiers should have corresponding display names"""
        tier_enum_members = set(APITier)
        tier_names_keys = set(TIER_NAMES.keys())

        assert tier_enum_members == tier_names_keys, (
            "Mismatch between APITier enum and TIER_NAMES"
        )

    def test_rate_limits_data_types(self):
        """Rate limits should have correct data types and reasonable values"""
        for tier, models in TIER_RATE_LIMITS.items():
            assert isinstance(tier, APITier)
            assert isinstance(models, dict)

            for model_name, rate_limits in models.items():
                assert isinstance(model_name, str)
                assert isinstance(rate_limits, RateLimits)
                assert rate_limits.requests_per_minute > 0
                assert rate_limits.tokens_per_minute > 0

    def test_model_capabilities_data_types(self):
        """Model capabilities should have correct data types and values"""
        for model_name, capabilities in MODEL_CAPABILITIES.items():
            assert isinstance(model_name, str)
            assert isinstance(capabilities, ModelCapabilities)
            assert isinstance(capabilities.supports_caching, bool)
            assert isinstance(capabilities.supports_multimodal, bool)
            assert capabilities.context_window > 0

    def test_tier_hierarchy_makes_sense(self):
        """Higher tiers should generally have higher rate limits"""
        # Compare FREE vs TIER_1 for overlapping models
        free_models = set(TIER_RATE_LIMITS[APITier.FREE].keys())
        tier1_models = set(TIER_RATE_LIMITS[APITier.TIER_1].keys())
        common_models = free_models.intersection(tier1_models)

        for model in common_models:
            free_limits = TIER_RATE_LIMITS[APITier.FREE][model]
            tier1_limits = TIER_RATE_LIMITS[APITier.TIER_1][model]

            # TIER_1 should have higher or equal limits than FREE
            assert tier1_limits.requests_per_minute >= free_limits.requests_per_minute
            assert tier1_limits.tokens_per_minute >= free_limits.tokens_per_minute


class TestConfigManagerInitialization:
    """Test ConfigManager initialization scenarios"""

    def test_default_initialization(self):
        """Should initialize with default tier and model"""
        with patch.object(ConfigManager, "_detect_tier", return_value=APITier.FREE):
            config = ConfigManager()

            assert config.tier == APITier.FREE
            assert config.model in TIER_RATE_LIMITS[APITier.FREE]

    def test_custom_tier_initialization(self):
        """Should initialize with specified tier and default model for that tier"""
        config = ConfigManager(tier=APITier.TIER_1)

        assert config.tier == APITier.TIER_1
        assert config.model in TIER_RATE_LIMITS[APITier.TIER_1]

    def test_custom_tier_and_model_initialization(self):
        """Should initialize with specified tier and model"""
        config = ConfigManager(tier=APITier.TIER_1, model="gemini-2.0-flash")

        assert config.tier == APITier.TIER_1
        assert config.model == "gemini-2.0-flash"

    def test_invalid_model_for_tier_raises_error(self):
        """Should raise error when model is not available in specified tier"""
        # gemini-2.5-pro-preview-06-05 is not available in FREE tier
        with pytest.raises(GeminiBatchError, match="not available in free tier"):
            ConfigManager(tier=APITier.FREE, model="gemini-2.5-pro-preview-06-05")

    def test_nonexistent_model_raises_error(self):
        """Should raise error for completely nonexistent model"""
        with pytest.raises(GeminiBatchError, match="not available"):
            ConfigManager(tier=APITier.TIER_1, model="nonexistent-model")

    def test_tier_detection_fallback(self):
        """Should fall back to FREE tier when detection fails"""
        config = ConfigManager()
        assert config.tier == APITier.FREE  # Default fallback


class TestModelLimitsRetrieval:
    """Test get_model_limits functionality"""

    def test_get_model_limits_valid_model(self):
        """Should return complete ModelLimits for valid model in tier"""
        config = ConfigManager(tier=APITier.TIER_1, model="gemini-2.0-flash")
        limits = config.get_model_limits("gemini-2.0-flash")

        assert isinstance(limits, ModelLimits)
        assert limits.requests_per_minute == 2_000  # TIER_1 value
        assert limits.tokens_per_minute == 4_000_000  # TIER_1 value
        assert limits.supports_caching is True  # From capabilities
        assert limits.supports_multimodal is True  # From capabilities
        assert limits.context_window == 1_000_000  # From capabilities

    def test_get_model_limits_model_not_in_tier(self):
        """Should return None for model not available in current tier"""
        config = ConfigManager(tier=APITier.FREE)
        limits = config.get_model_limits("gemini-2.5-pro-preview-06-05")  # Not in FREE

        assert limits is None

    def test_get_model_limits_nonexistent_model(self):
        """Should return None for completely nonexistent model"""
        config = ConfigManager(tier=APITier.TIER_1)
        limits = config.get_model_limits("fake-model")

        assert limits is None

    def test_get_model_limits_different_tiers_same_model(self):
        """Should return different rate limits for same model in different tiers"""
        free_config = ConfigManager(tier=APITier.FREE)
        tier1_config = ConfigManager(tier=APITier.TIER_1)

        free_limits = free_config.get_model_limits("gemini-2.0-flash")
        tier1_limits = tier1_config.get_model_limits("gemini-2.0-flash")

        # Rate limits should differ
        assert free_limits.requests_per_minute != tier1_limits.requests_per_minute
        assert free_limits.tokens_per_minute != tier1_limits.tokens_per_minute

        # But capabilities should be identical
        assert free_limits.supports_caching == tier1_limits.supports_caching
        assert free_limits.supports_multimodal == tier1_limits.supports_multimodal
        assert free_limits.context_window == tier1_limits.context_window


class TestRateLimiterConfiguration:
    """Test rate limiter configuration generation"""

    def test_get_rate_limiter_config_valid_model(self):
        """Should return rate limiter config for valid model"""
        config = ConfigManager(tier=APITier.TIER_1, model="gemini-2.0-flash")
        rate_config = config.get_rate_limiter_config("gemini-2.0-flash")

        assert isinstance(rate_config, dict)
        assert "requests_per_minute" in rate_config
        assert "tokens_per_minute" in rate_config
        assert rate_config["requests_per_minute"] == 2_000
        assert rate_config["tokens_per_minute"] == 4_000_000

    def test_get_rate_limiter_config_invalid_model_raises_error(self):
        """Should raise error for model not available in tier"""
        config = ConfigManager(tier=APITier.FREE)

        with pytest.raises(GeminiBatchError, match="not available in free tier"):
            config.get_rate_limiter_config("gemini-2.5-pro-preview-06-05")

    def test_get_rate_limiter_config_includes_available_models_in_error(self):
        """Error message should include list of available models"""
        config = ConfigManager(tier=APITier.FREE)

        with pytest.raises(GeminiBatchError) as exc_info:
            config.get_rate_limiter_config("invalid-model")

        error_message = str(exc_info.value)
        assert "Available models:" in error_message
        # Should include at least one model from FREE tier
        assert "gemini-2.0-flash" in error_message


class TestOptimizedMethods:
    """Test the optimized methods for efficiency"""

    def test_get_default_model_returns_valid_model(self):
        """Should return a valid model for each tier"""
        for tier in APITier:
            if tier in TIER_RATE_LIMITS:  # Skip tiers without models
                config = ConfigManager(tier=tier)
                default_model = config._get_default_model()

                assert default_model in TIER_RATE_LIMITS[tier]
                assert isinstance(default_model, str)
                assert len(default_model) > 0

    def test_get_default_model_fallback_behavior(self):
        """Should provide fallback when tier has no models"""
        config = ConfigManager(tier=APITier.FREE)

        # Mock empty tier to test fallback
        with patch.dict(TIER_RATE_LIMITS, {APITier.FREE: {}}):
            default_model = config._get_default_model()
            assert default_model == "gemini-2.0-flash"  # Fallback value

    def test_get_available_models_caching_behavior(self):
        """Should cache available models list and reuse it"""
        config = ConfigManager(tier=APITier.TIER_1)

        # First call should compute the list
        models1 = config._get_available_models()
        assert isinstance(models1, list)
        assert len(models1) > 0

        # Second call should return cached version
        models2 = config._get_available_models()
        assert models1 is models2  # Same object reference (cached)

        # Should contain expected models for TIER_1
        expected_models = list(TIER_RATE_LIMITS[APITier.TIER_1].keys())
        assert set(models1) == set(expected_models)

    def test_get_available_models_different_instances_separate_caches(self):
        """Different ConfigManager instances should have separate caches"""
        config1 = ConfigManager(tier=APITier.FREE)
        config2 = ConfigManager(tier=APITier.TIER_1)

        models1 = config1._get_available_models()
        models2 = config2._get_available_models()

        # Should have different models for different tiers
        assert set(models1) != set(models2)
        # Should be different objects
        assert models1 is not models2

    def test_cached_available_models_used_in_error_messages(self):
        """Error messages should use cached available models"""
        config = ConfigManager(tier=APITier.FREE)

        # Prime the cache
        cached_models = config._get_available_models()

        # Trigger error that uses _get_available_models
        with pytest.raises(GeminiBatchError) as exc_info:
            config.get_rate_limiter_config("invalid-model")

        error_message = str(exc_info.value)

        # Should contain models from our cached list
        for model in cached_models[:2]:  # Check first couple models
            assert model in error_message


class TestTierNameFunctionality:
    """Test tier name retrieval"""

    def test_get_tier_name_all_tiers(self):
        """Should return proper display names for all tiers"""
        expected_names = {
            APITier.FREE: "Free Tier",
            APITier.TIER_1: "Tier 1 (Billing Enabled)",
            APITier.TIER_2: "Tier 2",
            APITier.TIER_3: "Tier 3",
        }

        for tier, expected_name in expected_names.items():
            config = ConfigManager(tier=tier)
            assert config.get_tier_name() == expected_name

    def test_get_tier_name_unknown_tier_fallback(self):
        """Should return fallback name for unknown tiers"""
        config = ConfigManager(tier=APITier.FREE)

        # Mock an unknown tier
        config.tier = "unknown_tier"
        assert config.get_tier_name() == "Unknown Tier"


class TestConfigManagerEdgeCases:
    """Test edge cases and error conditions"""

    def test_model_selection_with_missing_capabilities(self):
        """Should handle gracefully when model has rate limits but no capabilities"""
        config = ConfigManager(tier=APITier.TIER_1)

        # Mock a scenario where model exists in tier but not in capabilities
        with patch.dict(
            TIER_RATE_LIMITS,
            {APITier.TIER_1: {"missing-capabilities-model": RateLimits(100, 1000)}},
        ):
            limits = config.get_model_limits("missing-capabilities-model")
            assert limits is None  # Should return None when capabilities missing

    def test_model_selection_with_missing_rate_limits(self):
        """Should handle gracefully when model has capabilities but no rate limits"""
        config = ConfigManager(tier=APITier.TIER_1)

        # Mock a scenario where model exists in capabilities but not in current tier
        with patch.dict(
            MODEL_CAPABILITIES,
            {"orphaned-model": ModelCapabilities(True, True, 1000000)},
        ):
            limits = config.get_model_limits("orphaned-model")
            assert limits is None  # Should return None when rate limits missing

    def test_initialization_validation_happens_after_model_selection(self):
        """Should validate model availability after default model selection"""
        # This tests that validation occurs even with default model selection
        with patch.object(
            ConfigManager, "_get_default_model", return_value="invalid-model"
        ), patch.object(
            ConfigManager, "_get_model_from_env", return_value=None
        ), pytest.raises(GeminiBatchError, match="not available"):
            ConfigManager(tier=APITier.FREE)

    def test_tier_with_no_models_handles_gracefully(self):
        """Should handle tier with no available models gracefully"""
        # Create a config with empty tier (edge case)
        with patch.dict(TIER_RATE_LIMITS, {APITier.FREE: {}}), pytest.raises(
            GeminiBatchError, match="not available"
        ):
            # Should fail gracefully when trying to get default from empty tier
            ConfigManager(tier=APITier.FREE)  # No explicit model with empty tier


class TestConfigManagerPracticalScenarios:
    """Test realistic usage scenarios"""

    def test_multi_tier_deployment_scenario(self):
        """Test scenario where application supports multiple tiers"""
        # Simulate checking capabilities across tiers
        tiers_to_test = [APITier.FREE, APITier.TIER_1, APITier.TIER_2]
        model_to_test = "gemini-2.0-flash"

        capabilities_by_tier = {}
        for tier in tiers_to_test:
            config = ConfigManager(tier=tier, model=model_to_test)
            limits = config.get_model_limits(model_to_test)
            capabilities_by_tier[tier] = limits

        # Verify all tiers have the model
        assert all(limits is not None for limits in capabilities_by_tier.values())

        # Verify rate limits increase with tier
        free_limits = capabilities_by_tier[APITier.FREE]
        tier1_limits = capabilities_by_tier[APITier.TIER_1]
        tier2_limits = capabilities_by_tier[APITier.TIER_2]

        assert free_limits.requests_per_minute <= tier1_limits.requests_per_minute
        assert tier1_limits.requests_per_minute <= tier2_limits.requests_per_minute

    def test_model_availability_check_scenario(self):
        """Test scenario where app checks model availability before use"""
        config = ConfigManager(tier=APITier.FREE)

        # Test models that should be available
        available_models = ["gemini-2.0-flash", "gemini-1.5-flash"]
        for model in available_models:
            limits = config.get_model_limits(model)
            assert limits is not None, f"Model {model} should be available in FREE tier"

        # Test models that should NOT be available in FREE tier
        unavailable_models = ["gemini-2.5-pro-preview-06-05"]
        for model in unavailable_models:
            limits = config.get_model_limits(model)
            assert limits is None, f"Model {model} should NOT be available in FREE tier"

    def test_rate_limiter_configuration_scenario(self):
        """Test scenario where rate limiter needs configuration"""
        config = ConfigManager(tier=APITier.TIER_1, model="gemini-2.0-flash")

        # Get rate limiter config
        rate_config = config.get_rate_limiter_config("gemini-2.0-flash")

        # Verify it has the expected structure for rate limiter
        assert isinstance(rate_config, dict)
        assert "requests_per_minute" in rate_config
        assert "tokens_per_minute" in rate_config
        assert isinstance(rate_config["requests_per_minute"], int)
        assert isinstance(rate_config["tokens_per_minute"], int)

        # Verify values are reasonable
        assert rate_config["requests_per_minute"] > 0
        assert rate_config["tokens_per_minute"] > 0

    def test_error_handling_in_production_scenario(self):
        """Test error handling scenarios that might occur in production"""
        config = ConfigManager(tier=APITier.FREE)

        # Test error message quality for debugging
        with pytest.raises(GeminiBatchError) as exc_info:
            config.get_rate_limiter_config("nonexistent-model")

        error_msg = str(exc_info.value)

        # Error should be informative for debugging
        assert "nonexistent-model" in error_msg
        assert "not available" in error_msg
        assert "free tier" in error_msg.lower()
        assert "Available models:" in error_msg

        # Should suggest valid alternatives
        assert any(
            model in error_msg for model in ["gemini-2.0-flash", "gemini-1.5-flash"]
        )

    def test_configuration_consistency_across_instances(self):
        """Test that configuration remains consistent across multiple instances"""
        # Create multiple instances with same parameters
        configs = [
            ConfigManager(tier=APITier.TIER_1, model="gemini-2.0-flash")
            for _ in range(3)
        ]

        # All should have identical configuration
        reference_limits = configs[0].get_model_limits("gemini-2.0-flash")

        for config in configs[1:]:
            limits = config.get_model_limits("gemini-2.0-flash")
            assert limits.requests_per_minute == reference_limits.requests_per_minute
            assert limits.tokens_per_minute == reference_limits.tokens_per_minute
            assert limits.supports_caching == reference_limits.supports_caching
            assert limits.supports_multimodal == reference_limits.supports_multimodal
            assert limits.context_window == reference_limits.context_window

    def test_memory_efficiency_of_caching(self):
        """Test that caching doesn't create excessive memory overhead"""
        config = ConfigManager(tier=APITier.TIER_1)

        # Prime the cache
        models1 = config._get_available_models()

        # Multiple calls should return same object (memory efficient)
        models2 = config._get_available_models()
        models3 = config._get_available_models()

        assert models1 is models2 is models3

        # Verify the cache attribute exists
        assert hasattr(config, "_cached_available_models")
        assert config._cached_available_models is models1
