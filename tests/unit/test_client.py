from unittest.mock import patch

import pytest

from gemini_batch.client import GeminiClient
from gemini_batch.config import APITier, ConfigManager
from gemini_batch.exceptions import APIError, MissingKeyError, NetworkError
from tests.fixtures.api_responses import SAMPLE_RESPONSES


class TestGeminiClientInitialization:
    """Test GeminiClient initialization behavior"""

    def test_initializes_with_api_key(self, mock_genai_client):
        """Should initialize successfully with API key and set defaults"""
        client = GeminiClient(api_key="test_key_123456789012345678901234567890")
        assert client.model_name == "gemini-2.0-flash"  # default
        assert client.enable_caching is False
        assert client.rate_limit_requests == 15
        assert client.rate_limit_window == 60

    def test_uses_environment_variables(self, mock_env, mock_genai_client):
        """Should read API key and model from environment variables"""
        client = GeminiClient()
        assert client.model_name == "gemini-2.0-flash"

    def test_custom_model_name(self, mock_genai_client):
        """Should accept custom model name"""
        # Use a real model name that exists in the free tier
        client = GeminiClient(
            api_key="test_key_123456789012345678901234567890",
            model_name="gemini-1.5-flash",
        )
        assert client.model_name == "gemini-1.5-flash"

    def test_raises_on_missing_api_key(self, monkeypatch):
        """Should raise MissingKeyError when no API key is provided"""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(MissingKeyError, match="API key required"):
            GeminiClient()

    def test_warns_when_explicit_parameters_override_config_manager(
        self, mock_genai_client
    ):
        """Should warn when explicit parameters are provided alongside ConfigManager"""
        config = ConfigManager.for_testing(tier=APITier.FREE, model="gemini-2.0-flash")

        with patch("warnings.warn") as mock_warn:
            client = GeminiClient(
                config_manager=config,
                api_key="override_key_123456789012345678901234567890",
                model_name="override-model",
            )

            # Should warn about parameter override
            mock_warn.assert_called_once_with(
                "Explicit parameters override ConfigManager values. "
                "Consider using ConfigManager exclusively for cleaner configuration.",
                stacklevel=2,
            )

            # Should use the explicit parameters over config
            assert client.api_key == "override_key_123456789012345678901234567890"
            assert client.model_name == "override-model"

    def test_warns_when_tier_provided_with_config_manager(self, mock_genai_client):
        """Should warn when tier parameter is provided alongside ConfigManager"""
        config = ConfigManager.for_testing(tier=APITier.FREE, model="gemini-2.0-flash")

        with patch("warnings.warn") as mock_warn:
            client = GeminiClient(
                config_manager=config,
                tier=APITier.TIER_1,  # This should trigger warning
            )

            mock_warn.assert_called_once_with(
                "Explicit parameters override ConfigManager values. "
                "Consider using ConfigManager exclusively for cleaner configuration.",
                stacklevel=2,
            )

    def test_no_warning_when_only_config_manager_provided(self, mock_genai_client):
        """Should not warn when only ConfigManager is provided without explicit parameters"""
        config = ConfigManager.for_testing(tier=APITier.FREE, model="gemini-2.0-flash")

        with patch("warnings.warn") as mock_warn:
            client = GeminiClient(config_manager=config)

            # Should not call warn
            mock_warn.assert_not_called()

            # Should use config values
            assert client.api_key == config.api_key
            assert client.model_name == config.model


class TestGeminiClientFactoryMethods:
    """Test factory method functionality for GeminiClient"""

    def test_from_config_factory_method(self, mock_genai_client):
        """Should create client from ConfigManager using factory method"""
        config = ConfigManager.for_testing(
            tier=APITier.TIER_1, model="gemini-2.0-flash"
        )

        client = GeminiClient.from_config(config)

        assert client.config is config
        assert client.api_key == config.api_key
        assert client.model_name == config.model

    def test_from_config_with_additional_kwargs(self, mock_genai_client):
        """Should create client from ConfigManager and accept additional kwargs"""
        config = ConfigManager.for_testing(
            tier=APITier.TIER_1, model="gemini-2.0-flash"
        )

        client = GeminiClient.from_config(config, enable_caching=True)

        assert client.config is config
        assert client.enable_caching is True

    def test_from_env_factory_method(self, mock_env, mock_genai_client):
        """Should create client from environment variables using factory method"""
        with patch.object(ConfigManager, "from_env") as mock_from_env:
            config = ConfigManager.for_testing(
                tier=APITier.FREE, model="gemini-2.0-flash"
            )
            mock_from_env.return_value = config

            client = GeminiClient.from_env()

            mock_from_env.assert_called_once()
            assert client.config is config

    def test_from_env_with_additional_kwargs(self, mock_env, mock_genai_client):
        """Should create client from environment and accept additional kwargs"""
        with patch.object(ConfigManager, "from_env") as mock_from_env:
            config = ConfigManager.for_testing(
                tier=APITier.FREE, model="gemini-2.0-flash"
            )
            mock_from_env.return_value = config

            client = GeminiClient.from_env(enable_caching=True)

            assert client.enable_caching is True


class TestGeminiClientRateLimitingSetup:
    """Test rate limiting setup and fallback behavior"""

    def test_rate_limiting_setup_success(self, mock_genai_client):
        """Should set up rate limiting from config successfully"""
        config = ConfigManager.for_testing(
            tier=APITier.TIER_1, model="gemini-2.0-flash"
        )

        client = GeminiClient(config_manager=config)

        # Should use TIER_1 rate limits for gemini-2.0-flash
        assert client.rate_limit_requests == 2_000
        assert client.rate_limit_tokens == 4_000_000
        assert client.rate_limit_window == 60

    def test_rate_limiting_setup_fallback_on_config_error(self, mock_genai_client):
        """Should fall back to conservative defaults when config setup fails"""
        config = ConfigManager.for_testing(tier=APITier.FREE, model="gemini-2.0-flash")

        # Mock get_rate_limiter_config to raise an exception
        with patch.object(
            config, "get_rate_limiter_config", side_effect=Exception("Config error")
        ):
            client = GeminiClient(config_manager=config)

            # Should fall back to conservative defaults
            assert client.rate_limit_requests == 15
            assert client.rate_limit_tokens == 250_000
            assert client.rate_limit_window == 60

    def test_rate_limiting_setup_fallback_on_missing_model(self, mock_genai_client):
        """Should fall back to defaults when model is not found in config"""
        # Create a valid config first, then mock the get_rate_limiter_config to fail
        config = ConfigManager.for_testing(tier=APITier.FREE, model="gemini-2.0-flash")

        # Mock get_rate_limiter_config to raise an exception (simulating missing model)
        with patch.object(
            config, "get_rate_limiter_config", side_effect=Exception("Model not found")
        ):
            client = GeminiClient(config_manager=config)

            # Should fall back to conservative defaults
            assert client.rate_limit_requests == 15
            assert client.rate_limit_tokens == 250_000


class TestGeminiClientConfigSummary:
    """Test configuration summary functionality"""

    def test_get_config_summary_complete(self, mock_genai_client):
        """Should return complete configuration summary"""
        config = ConfigManager.for_testing(
            tier=APITier.TIER_1, model="gemini-2.0-flash"
        )
        client = GeminiClient(config_manager=config, enable_caching=True)

        summary = client.get_config_summary()

        # Should include config manager summary
        assert "tier" in summary
        assert "tier_name" in summary
        assert "model" in summary
        assert "api_key_present" in summary

        # Should include client-specific information
        assert summary["client_model_name"] == "gemini-2.0-flash"
        assert summary["rate_limit_requests"] == 2_000
        assert summary["rate_limit_tokens"] == 4_000_000
        assert summary["enable_caching"] is True

    def test_get_config_summary_with_missing_attributes(self, mock_genai_client):
        """Should handle missing rate_limit_tokens attribute gracefully"""
        config = ConfigManager.for_testing(tier=APITier.FREE, model="gemini-2.0-flash")
        client = GeminiClient(config_manager=config)

        # Remove rate_limit_tokens to test fallback
        delattr(client, "rate_limit_tokens")

        summary = client.get_config_summary()

        assert summary["rate_limit_tokens"] == "unknown"


class TestGeminiClientContentGeneration:
    """Test content generation functionality"""

    def test_generates_content_successfully(self, gemini_client, mock_genai_client):
        """Should generate content and return text response"""
        mock_genai_client.models.generate_content.return_value = SAMPLE_RESPONSES[
            "simple_answer"
        ]

        result = gemini_client.generate_content("Test prompt")

        assert result == SAMPLE_RESPONSES["simple_answer"].text
        # Verify the API was called with correct parameters
        mock_genai_client.models.generate_content.assert_called_once_with(
            model="gemini-2.0-flash", contents="Test prompt"
        )

    def test_generates_content_with_system_instruction(
        self, gemini_client, mock_genai_client
    ):
        """Should include system instruction when provided"""
        mock_genai_client.models.generate_content.return_value = SAMPLE_RESPONSES[
            "simple_answer"
        ]

        result = gemini_client.generate_content(
            "Test prompt", system_instruction="Be helpful and concise"
        )

        # Verify system instruction was passed
        call_args = mock_genai_client.models.generate_content.call_args
        assert call_args[1]["model"] == "gemini-2.0-flash"
        assert call_args[1]["contents"] == "Test prompt"
        assert "config" in call_args[1]

    def test_returns_usage_metrics_when_requested(
        self, gemini_client, mock_genai_client
    ):
        """Should return usage metrics when return_usage=True"""
        mock_genai_client.models.generate_content.return_value = SAMPLE_RESPONSES[
            "simple_answer"
        ]

        result = gemini_client.generate_content("Test prompt", return_usage=True)

        assert isinstance(result, dict)
        assert "text" in result
        assert "usage" in result
        assert result["text"] == SAMPLE_RESPONSES["simple_answer"].text
        assert result["usage"]["prompt_tokens"] == 150
        assert result["usage"]["output_tokens"] == 75

    def test_converts_api_exceptions_to_api_error(
        self, gemini_client, mock_genai_client
    ):
        """Should convert generic API exceptions to APIError"""
        mock_genai_client.models.generate_content.side_effect = Exception(
            "Generic API error"
        )

        with pytest.raises(APIError, match="API call failed: Generic API error"):
            gemini_client.generate_content("Test prompt")

    def test_converts_network_exceptions_to_network_error(
        self, gemini_client, mock_genai_client
    ):
        """Should convert connection errors to NetworkError"""
        mock_genai_client.models.generate_content.side_effect = ConnectionError(
            "Network down"
        )

        with pytest.raises(NetworkError, match="Network connection failed"):
            gemini_client.generate_content("Test prompt")


class TestGeminiClientRetryLogic:
    """Test API retry logic and rate limiting behavior"""

    def test_retry_logic_on_rate_limit_errors(self, gemini_client, mock_genai_client):
        """Should retry on rate limit errors with exponential backoff"""
        # First call fails with rate limit, second succeeds
        mock_genai_client.models.generate_content.side_effect = [
            Exception("Rate limit exceeded"),
            SAMPLE_RESPONSES["simple_answer"],
        ]

        with patch("time.sleep") as mock_sleep:
            result = gemini_client.generate_content("Test prompt")

            # Should have retried and succeeded
            assert result == SAMPLE_RESPONSES["simple_answer"].text
            assert mock_genai_client.models.generate_content.call_count == 2

            # Should have slept with exponential backoff (5 seconds for first retry)
            mock_sleep.assert_called_with(5)

    def test_retry_logic_on_quota_errors(self, gemini_client, mock_genai_client):
        """Should retry on quota errors"""
        mock_genai_client.models.generate_content.side_effect = [
            Exception("Quota exceeded"),
            SAMPLE_RESPONSES["simple_answer"],
        ]

        with patch("time.sleep") as mock_sleep:
            result = gemini_client.generate_content("Test prompt")

            assert result == SAMPLE_RESPONSES["simple_answer"].text
            assert mock_genai_client.models.generate_content.call_count == 2
            mock_sleep.assert_called_once()

    def test_retry_logic_on_429_errors(self, gemini_client, mock_genai_client):
        """Should retry on HTTP 429 errors"""
        mock_genai_client.models.generate_content.side_effect = [
            Exception("HTTP 429 Too Many Requests"),
            SAMPLE_RESPONSES["simple_answer"],
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
            # Should use shorter delay for non-rate-limit errors (1 second for first retry)
            mock_sleep.assert_called_with(1)

    def test_retry_exhaustion_raises_final_error(
        self, gemini_client, mock_genai_client
    ):
        """Should raise the final error when all retries are exhausted"""
        mock_genai_client.models.generate_content.side_effect = Exception(
            "Persistent error"
        )

        with patch("time.sleep"):
            with pytest.raises(APIError, match="API call failed: Persistent error"):
                gemini_client.generate_content("Test prompt")

            # Should have tried 3 times total (1 initial + 2 retries)
            assert mock_genai_client.models.generate_content.call_count == 3

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
            # First retry: 2^0 * 5 = 5 seconds
            # Second retry: 2^1 * 5 = 10 seconds
            expected_calls = [((5,), {}), ((10,), {})]
            assert mock_sleep.call_args_list == expected_calls


class TestGeminiClientRateLimiting:
    """Test rate limiting functionality"""

    def test_tracks_request_timestamps(self, gemini_client):
        """Should track timestamps of API requests"""
        initial_count = len(gemini_client.request_timestamps)

        with patch("time.time", return_value=1000.0):
            gemini_client._wait_for_rate_limit()

        assert len(gemini_client.request_timestamps) == initial_count + 1
        assert gemini_client.request_timestamps[-1] == 1000.0

    def test_sleeps_when_rate_limit_exceeded(self, gemini_client):
        """Should sleep when rate limit is exceeded"""
        # Fill up to rate limit
        current_time = 1000.0
        for _ in range(gemini_client.rate_limit_requests):
            gemini_client.request_timestamps.append(current_time)

        with patch("time.time", return_value=current_time):
            with patch("time.sleep") as mock_sleep:
                gemini_client._wait_for_rate_limit()
                mock_sleep.assert_called_once()
                # Should sleep for approximately the rate limit window + 1
                sleep_duration = mock_sleep.call_args[0][0]
                assert sleep_duration > 60  # Should be rate_limit_window + 1

    def test_cleans_old_timestamps(self, gemini_client):
        """Should remove timestamps older than rate limit window"""
        current_time = 1000.0
        old_time = current_time - gemini_client.rate_limit_window - 10

        # Add old timestamps
        gemini_client.request_timestamps.extend([old_time, old_time + 5])
        initial_count = len(gemini_client.request_timestamps)

        with patch("time.time", return_value=current_time):
            gemini_client._wait_for_rate_limit()

        # Old timestamps should be removed
        assert len(gemini_client.request_timestamps) < initial_count + 1


class TestGeminiClientBatchProcessing:
    """Test batch-specific functionality"""

    def test_creates_properly_formatted_batch_prompt(self, gemini_client):
        """Should create well-formatted batch prompt with all required elements"""
        content = "AI is transforming technology"
        questions = ["What is AI?", "How does it work?", "What are the benefits?"]

        prompt = gemini_client._create_batch_prompt(content, questions)

        # Verify content is included
        assert "AI is transforming technology" in prompt

        # Verify all questions are numbered and included
        assert "Question 1: What is AI?" in prompt
        assert "Question 2: How does it work?" in prompt
        assert "Question 3: What are the benefits?" in prompt

        # Verify answer format template is provided
        assert "Answer 1: [Your response]" in prompt
        assert "Answer 2: [Your response]" in prompt
        assert "Answer 3: [Your response]" in prompt

    def test_generates_batch_response(self, gemini_client, mock_genai_client):
        """Should generate batch response using properly formatted prompt"""
        content = "Test content"
        questions = ["Question 1?", "Question 2?"]

        mock_genai_client.models.generate_content.return_value = SAMPLE_RESPONSES[
            "batch_answer"
        ]

        result = gemini_client.generate_batch(content, questions)

        assert result == SAMPLE_RESPONSES["batch_answer"].text

        # Verify the underlying generate_content was called with batch prompt
        mock_genai_client.models.generate_content.assert_called_once()
        call_args = mock_genai_client.models.generate_content.call_args
        batch_prompt = call_args[1]["contents"]
        assert "Test content" in batch_prompt
        assert "Question 1:" in batch_prompt
        assert "Question 2:" in batch_prompt

    def test_generates_batch_with_usage_metrics(self, gemini_client, mock_genai_client):
        """Should return usage metrics for batch processing when requested"""
        mock_genai_client.models.generate_content.return_value = SAMPLE_RESPONSES[
            "batch_answer"
        ]

        result = gemini_client.generate_batch("content", ["Q1?"], return_usage=True)

        assert isinstance(result, dict)
        assert "text" in result
        assert "usage" in result
