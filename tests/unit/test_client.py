from unittest.mock import patch

import pytest

from gemini_batch.client import GeminiClient
from gemini_batch.exceptions import APIError, MissingKeyError, NetworkError
from tests.fixtures.api_responses import SAMPLE_RESPONSES


class TestGeminiClientInitialization:
    """Test GeminiClient initialization behavior"""

    def test_initializes_with_api_key(self, mock_genai_client):
        """Should initialize successfully with API key and set defaults"""
        client = GeminiClient(api_key="test_key")
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
        client = GeminiClient(api_key="test_key", model_name="custom-model")
        assert client.model_name == "custom-model"

    def test_raises_on_missing_api_key(self, monkeypatch):
        """Should raise MissingKeyError when no API key is provided"""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(MissingKeyError, match="API key required"):
            GeminiClient()


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
