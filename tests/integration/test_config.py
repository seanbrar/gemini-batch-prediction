from unittest.mock import patch

import pytest

from gemini_batch.config import APITier, ConfigManager
from gemini_batch.exceptions import GeminiBatchError


class TestConfigManagerIntegration:
    """Test ConfigManager integration with environment and deployment scenarios"""

    def test_environment_fallback_chain(self):
        """Should handle complete environment fallback chain correctly"""
        # Test with mixed environment setup
        with patch.dict(
            "os.environ",
            {
                "GEMINI_API_KEY": "fallback_key_123456789012345678901234567890",  # Primary key
                "GEMINI_MODEL": "gemini-1.5-flash",  # Direct
                # GEMINI_TIER missing - should detect from key
            },
            clear=False,
        ):
            config = ConfigManager.from_env()

            # Should use the key
            assert config.api_key == "fallback_key_123456789012345678901234567890"
            assert config.model == "gemini-1.5-flash"
            # Tier should be detected or default to FREE
            assert config.tier in [
                APITier.FREE,
                APITier.TIER_1,
                APITier.TIER_2,
                APITier.TIER_3,
            ]

    def test_complex_environment_scenario(self):
        """Should handle complex real-world environment scenarios"""
        # Test deployment with overrides and fallbacks
        with patch.dict(
            "os.environ",
            {
                "GEMINI_API_KEY": "primary_key_123456789012345678901234567890",
                "GOOGLE_API_KEY": "fallback_key_123456789012345678901234567890",
                "GEMINI_TIER": "tier_2",
                "GEMINI_MODEL": "gemini-2.0-flash",
                "APP_ENV": "production",  # Additional env var that shouldn't interfere
            },
            clear=False,
        ):
            config = ConfigManager.from_env()

            # Should use primary values
            assert config.api_key == "primary_key_123456789012345678901234567890"
            assert config.tier == APITier.TIER_2
            assert config.model == "gemini-2.0-flash"

            # Should be able to get working configuration
            rate_config = config.get_rate_limiter_config("gemini-2.0-flash")
            assert rate_config["requests_per_minute"] > 100

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
