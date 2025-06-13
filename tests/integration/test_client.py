from unittest.mock import patch

from gemini_batch.client import GeminiClient
from gemini_batch.config import APITier, ConfigManager
from tests.fixtures.api_responses import SAMPLE_RESPONSES


class TestGeminiClientIntegration:
    """Test GeminiClient integration with other components"""

    def test_end_to_end_batch_processing_workflow(self, mock_genai_client):
        """Should handle complete batch processing workflow end-to-end with ConfigManager integration"""
        # Create client using ConfigManager factory method
        config = ConfigManager.for_testing(
            tier=APITier.TIER_1, model="gemini-2.0-flash"
        )
        client = GeminiClient.from_config(config)

        # Setup realistic batch response
        batch_response = type(SAMPLE_RESPONSES["batch_answer"])(
            text="Answer 1: First response\nAnswer 2: Second response",
            usage_metadata=SAMPLE_RESPONSES["batch_answer"].usage_metadata,
        )
        mock_genai_client.models.generate_content.return_value = batch_response

        # Process batch - this tests integration of:
        # - ConfigManager providing proper configuration
        # - GeminiClient using config for model selection and rate limiting
        # - Batch prompt creation and processing
        # - Response handling and parsing
        response = client.generate_batch("Test content", ["Q1?", "Q2?"])

        # Verify response structure and content
        assert isinstance(response, str)
        assert "Answer 1:" in response
        assert "Answer 2:" in response

        # Verify integration with ConfigManager worked properly
        assert client.model_name == "gemini-2.0-flash"
        assert client.config is config
        assert client.api_key == config.api_key

        # Verify rate limiting configuration from config was applied
        assert client.rate_limit_requests == 2_000  # TIER_1 limits
        assert client.rate_limit_tokens == 4_000_000

    def test_factory_method_integration_with_environment(
        self, mock_env, mock_genai_client
    ):
        """Should integrate with environment configuration end-to-end"""
        # Test the from_env factory method working with environment variables
        with patch.dict(
            "os.environ",
            {
                "GEMINI_API_KEY": "integration_test_key_123456789012345678901234567890",
                "GEMINI_MODEL": "gemini-1.5-flash",
                "GEMINI_TIER": "tier_2",
            },
            clear=False,
        ):
            # Create client from environment
            client = GeminiClient.from_env()

            # Verify environment integration worked
            assert (
                client.api_key == "integration_test_key_123456789012345678901234567890"
            )
            assert client.model_name == "gemini-1.5-flash"
            assert client.config.tier == APITier.TIER_2

            # Verify rate limiting was configured from tier
            assert client.rate_limit_requests >= 2_000  # TIER_2 should have high limits
