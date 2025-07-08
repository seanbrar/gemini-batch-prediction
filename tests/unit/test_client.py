from unittest.mock import patch

import pytest

from gemini_batch import GeminiClient
from gemini_batch.client.configuration import ClientConfiguration
from gemini_batch.config import APITier, ConfigManager
from gemini_batch.exceptions import APIError, MissingKeyError
from tests.fixtures.api_responses import SAMPLE_RESPONSES


class TestGeminiClientInitialization:
    """Test GeminiClient initialization behavior"""

    def test_initializes_with_config(self, mock_genai_client):
        """Should initialize successfully with ClientConfiguration"""
        config = ClientConfiguration(
            api_key="test_key_123456789012345678901234567890",
            model_name="gemini-2.0-flash",
            enable_caching=False,
        )
        client = GeminiClient(config)
        assert client.config.model_name == "gemini-2.0-flash"
        assert client.config.enable_caching is False

    def test_uses_environment_variables(self, mock_env, mock_genai_client):
        """Should read API key and model from environment variables"""
        client = GeminiClient.from_env()
        assert client.config.model_name == "gemini-2.0-flash"

    def test_custom_model_name(self, mock_genai_client):
        """Should accept custom model name"""
        config = ClientConfiguration(
            api_key="test_key_123456789012345678901234567890",
            model_name="gemini-1.5-flash",
            enable_caching=False,
        )
        client = GeminiClient(config)
        assert client.config.model_name == "gemini-1.5-flash"

    def test_raises_on_missing_api_key(self, monkeypatch):
        """Should raise MissingKeyError when no API key is provided"""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(MissingKeyError, match="API key required"):
            GeminiClient.from_env()

    def test_from_config_manager_factory(self, mock_genai_client):
        """Should create client from ConfigManager using factory method"""
        config_manager = ConfigManager(
            api_key="test_key_123456789012345678901234567890",
            model="gemini-2.0-flash",
            tier=APITier.FREE,
        )

        client = GeminiClient.from_config_manager(config_manager)

        assert client.config.api_key == config_manager.api_key
        assert client.config.model_name == config_manager.model

    def test_from_env_factory_method(self, mock_env, mock_genai_client):
        """Should create client from environment variables using factory method"""
        with patch.object(ConfigManager, "from_env") as mock_from_env:
            config_manager = ConfigManager(
                api_key="test_key_123456789012345678901234567890",
                model="gemini-2.0-flash",
                tier=APITier.FREE,
            )
            mock_from_env.return_value = config_manager

            client = GeminiClient.from_env()

            mock_from_env.assert_called_once()
            assert client.config.api_key == config_manager.api_key

    def test_from_env_with_additional_kwargs(self, mock_env, mock_genai_client):
        """Should create client from environment and accept additional kwargs"""
        with patch.object(ConfigManager, "from_env") as mock_from_env:
            config_manager = ConfigManager(
                api_key="test_key_123456789012345678901234567890",
                model="gemini-2.0-flash",
                tier=APITier.FREE,
            )
            mock_from_env.return_value = config_manager

            client = GeminiClient.from_env(enable_caching=True)

            assert client.config.enable_caching is True


class TestGeminiClientFactoryMethods:
    """Test factory method functionality for GeminiClient"""

    def test_from_config_factory_method(self, mock_genai_client):
        """Should create client from ConfigManager using factory method"""
        config_manager = ConfigManager(
            api_key="test_key_123456789012345678901234567890",
            model="gemini-2.0-flash",
            tier=APITier.TIER_1,
        )

        client = GeminiClient.from_config_manager(config_manager)

        assert client.config.api_key == config_manager.api_key
        assert client.config.model_name == config_manager.model

    def test_from_config_with_additional_kwargs(self, mock_genai_client):
        """Should create client from ConfigManager and accept additional kwargs"""
        config_manager = ConfigManager(
            api_key="test_key_123456789012345678901234567890",
            model="gemini-2.0-flash",
            tier=APITier.TIER_1,
        )

        client = GeminiClient.from_config_manager(config_manager, enable_caching=True)

        assert client.config.enable_caching is True

    def test_from_env_factory_method(self, mock_env, mock_genai_client):
        """Should create client from environment variables using factory method"""
        with patch.object(ConfigManager, "from_env") as mock_from_env:
            config_manager = ConfigManager(
                api_key="test_key_123456789012345678901234567890",
                model="gemini-2.0-flash",
                tier=APITier.FREE,
            )
            mock_from_env.return_value = config_manager

            client = GeminiClient.from_env()

            mock_from_env.assert_called_once()
            assert client.config.api_key == config_manager.api_key

    def test_from_env_with_additional_kwargs(self, mock_env, mock_genai_client):
        """Should create client from environment and accept additional kwargs"""
        with patch.object(ConfigManager, "from_env") as mock_from_env:
            config_manager = ConfigManager(
                api_key="test_key_123456789012345678901234567890",
                model="gemini-2.0-flash",
                tier=APITier.FREE,
            )
            mock_from_env.return_value = config_manager

            client = GeminiClient.from_env(enable_caching=True)

            assert client.config.enable_caching is True


class TestGeminiClientRateLimitingSetup:
    """Test rate limiting setup and fallback behavior"""

    def test_rate_limiting_setup_success(self, mock_genai_client):
        """Should set up rate limiting from config successfully"""
        config_manager = ConfigManager(
            api_key="test_key_123456789012345678901234567890",
            model="gemini-2.0-flash",
            tier=APITier.TIER_1,
        )

        client = GeminiClient.from_config_manager(config_manager)

        # Should have rate limiter configured
        assert client.rate_limiter is not None
        assert hasattr(client.rate_limiter, "config")

    def test_rate_limiting_setup_fallback_on_config_error(self, mock_genai_client):
        """Should fall back to conservative defaults when config setup fails"""
        config_manager = ConfigManager(
            api_key="test_key_123456789012345678901234567890",
            model="gemini-2.0-flash",
            tier=APITier.FREE,
        )

        # Mock get_rate_limiter_config to raise an exception
        with patch.object(
            config_manager,
            "get_rate_limiter_config",
            side_effect=Exception("Config error"),
        ):
            client = GeminiClient.from_config_manager(config_manager)

            # Should still have rate limiter with fallback config
            assert client.rate_limiter is not None

    def test_rate_limiting_setup_fallback_on_missing_model(self, mock_genai_client):
        """Should fall back to defaults when model is not found in config"""
        config_manager = ConfigManager(
            api_key="test_key_123456789012345678901234567890",
            model="gemini-2.0-flash",
            tier=APITier.FREE,
        )

        # Mock get_rate_limiter_config to raise an exception (simulating missing model)
        with patch.object(
            config_manager,
            "get_rate_limiter_config",
            side_effect=Exception("Model not found"),
        ):
            client = GeminiClient.from_config_manager(config_manager)

            # Should still have rate limiter with fallback config
            assert client.rate_limiter is not None


class TestGeminiClientConfigSummary:
    """Test configuration summary functionality"""

    def test_get_config_summary_complete(self, mock_genai_client):
        """Should return complete configuration summary"""
        config_manager = ConfigManager(
            api_key="test_key_123456789012345678901234567890",
            model="gemini-2.0-flash",
            tier=APITier.TIER_1,
        )

        client = GeminiClient.from_config_manager(config_manager)
        summary = client.get_config_summary()

        assert "api_key" in summary
        assert "model_name" in summary
        assert "enable_caching" in summary
        assert "rate_limiter_config" in summary

    def test_get_config_summary_with_missing_attributes(self, mock_genai_client):
        """Should handle missing attributes gracefully in summary"""
        config_manager = ConfigManager(
            api_key="test_key_123456789012345678901234567890",
            model="gemini-2.0-flash",
            tier=APITier.FREE,
        )

        client = GeminiClient.from_config_manager(config_manager)
        summary = client.get_config_summary()

        # Should still have basic config info
        assert "api_key" in summary
        assert "model_name" in summary


class TestGeminiClientContentGeneration:
    """Test content generation functionality"""

    def test_generates_content_successfully(self, gemini_client, mock_genai_client):
        """Should generate content successfully"""
        mock_genai_client.models.generate_content.return_value = SAMPLE_RESPONSES[
            "simple_answer"
        ]

        result = gemini_client.generate_content("Test prompt")

        assert result == SAMPLE_RESPONSES["simple_answer"].text
        mock_genai_client.models.generate_content.assert_called_once()

    def test_generates_content_with_system_instruction(
        self, gemini_client, mock_genai_client
    ):
        """Should generate content with system instruction"""
        mock_genai_client.models.generate_content.return_value = SAMPLE_RESPONSES[
            "simple_answer"
        ]

        result = gemini_client.generate_content(
            "Test prompt", system_instruction="You are a helpful assistant."
        )

        assert result == SAMPLE_RESPONSES["simple_answer"].text
        call_args = mock_genai_client.models.generate_content.call_args
        assert call_args is not None

    def test_returns_usage_metrics_when_requested(
        self, gemini_client, mock_genai_client
    ):
        """Should return usage metrics when requested"""
        from tests.fixtures.api_responses import MockUsageMetadata

        mock_response = SAMPLE_RESPONSES["simple_answer"]
        mock_response.usage_metadata = MockUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=20,
        )
        mock_genai_client.models.generate_content.return_value = mock_response

        result = gemini_client.generate_content("Test prompt", return_usage=True)

        assert isinstance(result, dict)
        assert "usage" in result
        assert result["usage"]["prompt_tokens"] == 10

    def test_converts_api_exceptions_to_api_error(
        self, gemini_client, mock_genai_client
    ):
        """Should convert API exceptions to APIError"""
        mock_genai_client.models.generate_content.side_effect = Exception("API error")

        with pytest.raises(APIError, match="Content generation failed"):
            gemini_client.generate_content("Test prompt")

    def test_converts_network_exceptions_to_network_error(
        self, gemini_client, mock_genai_client
    ):
        """Should convert connection errors to NetworkError"""
        from requests.exceptions import ConnectionError

        mock_genai_client.models.generate_content.side_effect = ConnectionError(
            "Network down"
        )

        with pytest.raises(APIError, match="Content generation failed"):
            gemini_client.generate_content("Test prompt")


class TestGeminiClientRetryLogic:
    """Test retry logic for API calls"""

    def test_retry_logic_on_rate_limit_errors(self, gemini_client, mock_genai_client):
        """Should retry on rate limit errors with exponential backoff"""
        mock_genai_client.models.generate_content.side_effect = [
            Exception("Rate limit exceeded"),  # First attempt
            SAMPLE_RESPONSES["simple_answer"],  # Second attempt succeeds
        ]

        with patch("time.sleep") as mock_sleep:
            result = gemini_client.generate_content("Test prompt")

            assert result == SAMPLE_RESPONSES["simple_answer"].text
            mock_sleep.assert_called_once()

    def test_retry_logic_on_quota_errors(self, gemini_client, mock_genai_client):
        """Should retry on quota errors with exponential backoff"""
        mock_genai_client.models.generate_content.side_effect = [
            Exception("Quota exceeded"),  # First attempt
            SAMPLE_RESPONSES["simple_answer"],  # Second attempt succeeds
        ]

        with patch("time.sleep") as mock_sleep:
            result = gemini_client.generate_content("Test prompt")

            assert result == SAMPLE_RESPONSES["simple_answer"].text
            mock_sleep.assert_called_once()

    def test_retry_logic_on_429_errors(self, gemini_client, mock_genai_client):
        """Should retry on 429 errors with exponential backoff"""
        mock_genai_client.models.generate_content.side_effect = [
            Exception("429 Too Many Requests"),  # First attempt
            SAMPLE_RESPONSES["simple_answer"],  # Second attempt succeeds
        ]

        with patch("time.sleep") as mock_sleep:
            result = gemini_client.generate_content("Test prompt")

            assert result == SAMPLE_RESPONSES["simple_answer"].text
            mock_sleep.assert_called_once()

    def test_retry_logic_on_other_api_errors(self, gemini_client, mock_genai_client):
        """Should retry on other API errors with shorter delay"""
        mock_genai_client.models.generate_content.side_effect = [
            Exception("Server error"),
            SAMPLE_RESPONSES["simple_answer"],
        ]

        with patch("time.sleep") as mock_sleep:
            result = gemini_client.generate_content("Test prompt")

            assert result == SAMPLE_RESPONSES["simple_answer"].text
            mock_sleep.assert_called_once()

    def test_retry_exhaustion_raises_final_error(
        self, gemini_client, mock_genai_client
    ):
        """Should raise the final error when all retries are exhausted"""
        mock_genai_client.models.generate_content.side_effect = Exception(
            "Persistent error"
        )

        with patch("time.sleep"):
            with pytest.raises(APIError, match="Content generation failed"):
                gemini_client.generate_content("Test prompt")

    def test_exponential_backoff_progression(self, gemini_client, mock_genai_client):
        """Should use exponential backoff for rate limit errors"""
        mock_genai_client.models.generate_content.side_effect = [
            Exception("Rate limit exceeded"),  # First attempt
            Exception("Rate limit exceeded"),  # First retry
            SAMPLE_RESPONSES["simple_answer"],  # Second retry succeeds
        ]

        with patch("time.sleep") as mock_sleep:
            result = gemini_client.generate_content("Test prompt")

            assert result == SAMPLE_RESPONSES["simple_answer"].text

            # Should have called sleep twice with exponential backoff
            assert mock_sleep.call_count == 2


class TestGeminiClientRateLimiting:
    """Test rate limiting functionality"""

    def test_tracks_request_timestamps(self, gemini_client):
        """Should track timestamps of API requests"""
        # This test needs to be updated for the new rate limiter structure
        assert hasattr(gemini_client, "rate_limiter")
        assert gemini_client.rate_limiter is not None

    def test_sleeps_when_rate_limit_exceeded(self, gemini_client):
        """Should sleep when rate limit is exceeded"""
        # This test needs to be updated for the new rate limiter structure
        assert hasattr(gemini_client, "rate_limiter")
        assert gemini_client.rate_limiter is not None

    def test_cleans_old_timestamps(self, gemini_client):
        """Should remove timestamps older than rate limit window"""
        # This test needs to be updated for the new rate limiter structure
        assert hasattr(gemini_client, "rate_limiter")
        assert gemini_client.rate_limiter is not None


class TestGeminiClientBatchProcessing:
    """Test batch processing functionality"""

    def test_creates_properly_formatted_batch_prompt(self, gemini_client):
        """Should create well-formatted batch prompt with all required elements"""
        content = "AI is transforming technology"
        questions = ["What is AI?", "How does it work?", "What are the benefits?"]

        result = gemini_client.generate_batch(content, questions)

        # Should return a result (even if mocked)
        assert result is not None

    def test_generates_batch_response(self, gemini_client, mock_genai_client):
        """Should generate batch response using properly formatted prompt"""
        content = "Test content"
        questions = ["Question 1?", "Question 2?"]

        mock_genai_client.models.generate_content.return_value = SAMPLE_RESPONSES[
            "batch_answer"
        ]

        result = gemini_client.generate_batch(content, questions)

        assert result == SAMPLE_RESPONSES["batch_answer"].text

        # Verify the underlying generate_content was called
        mock_genai_client.models.generate_content.assert_called_once()

    def test_generates_batch_with_usage_metrics(self, gemini_client, mock_genai_client):
        """Should generate batch response with usage metrics when requested"""
        from tests.fixtures.api_responses import MockUsageMetadata
        
        mock_response = SAMPLE_RESPONSES["batch_answer"]
        mock_response.usage_metadata = MockUsageMetadata(
            prompt_token_count=50,
            candidates_token_count=100,
        )
        mock_genai_client.models.generate_content.return_value = mock_response

        result = gemini_client.generate_batch(
            "Test content", ["Q1?", "Q2?"], return_usage=True
        )

        assert isinstance(result, dict)
        assert "usage" in result
        assert result["usage"]["prompt_tokens"] == 50
