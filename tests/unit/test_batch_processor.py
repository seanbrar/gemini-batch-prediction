from unittest.mock import Mock, patch

import pytest

from gemini_batch import BatchProcessor
from gemini_batch.exceptions import APIError, MissingKeyError, NetworkError
from tests.fixtures.api_responses import SAMPLE_RESPONSES


class TestBatchProcessorInitialization:
    """Test BatchProcessor initialization behavior"""

    def test_creates_client_when_none_provided(self):
        """Should create a GeminiClient when no client is provided"""
        with patch("gemini_batch.batch_processor.GeminiClient") as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance

            processor = BatchProcessor()

            assert processor.client is mock_client_instance
            mock_client_class.assert_called_once_with(api_key=None)

    def test_uses_provided_client(self, mock_genai_client):
        """Should use the provided client instance"""
        from gemini_batch.client import GeminiClient

        mock_client = GeminiClient(api_key="test_key_123456789012345678901234567890")
        processor = BatchProcessor(client=mock_client)

        assert processor.client is mock_client

    def test_passes_kwargs_to_new_client(self):
        """Should pass initialization parameters to new GeminiClient"""
        with patch("gemini_batch.batch_processor.GeminiClient") as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance

            processor = BatchProcessor(
                api_key="test_key_123456789012345678901234567890",
                model_name="custom-model",
                enable_caching=True,
            )

            mock_client_class.assert_called_once_with(
                api_key="test_key_123456789012345678901234567890",
                model_name="custom-model",
                enable_caching=True,
            )

    def test_warns_when_client_and_kwargs_both_provided(self, mock_genai_client):
        """Should warn user when both client and kwargs are provided (kwargs ignored)"""
        from gemini_batch.client import GeminiClient

        mock_client = GeminiClient(api_key="test_key_123456789012345678901234567890")

        with patch("warnings.warn") as mock_warn:
            processor = BatchProcessor(
                api_key="ignored_key", client=mock_client, model_name="ignored_model"
            )

            mock_warn.assert_called_once_with(
                "api_key and client_kwargs ignored when client is provided",
                stacklevel=2,
            )
            assert processor.client is mock_client

    @pytest.mark.parametrize(
        "exception_type,api_key,expected_message",
        [
            (MissingKeyError, None, "API key required"),
            (MissingKeyError, "", "API key must be a non-empty string"),
            (
                NetworkError,
                "valid_key_123456789012345678901234567890",
                "Unable to connect",
            ),
        ],
    )
    def test_initialization_error_propagation(
        self, exception_type, api_key, expected_message
    ):
        """Should propagate appropriate errors from client initialization"""
        with patch("gemini_batch.batch_processor.GeminiClient") as mock_client_class:
            mock_client_class.side_effect = exception_type(expected_message)

            with pytest.raises(exception_type, match=expected_message):
                BatchProcessor(api_key=api_key)

    def test_handles_missing_key_error_with_client_kwargs(self):
        """Should propagate MissingKeyError even when additional client kwargs are provided"""
        with patch("gemini_batch.batch_processor.GeminiClient") as mock_client_class:
            mock_client_class.side_effect = MissingKeyError(
                "API key must be a non-empty string"
            )

            with pytest.raises(
                MissingKeyError, match="API key must be a non-empty string"
            ):
                BatchProcessor(
                    api_key="",  # Empty API key
                    model_name="gemini-2.0-flash",
                    enable_caching=True,
                )

            # Verify the full parameter set was passed
            mock_client_class.assert_called_once_with(
                api_key="", model_name="gemini-2.0-flash", enable_caching=True
            )

    def test_handles_network_error_with_complex_initialization(self):
        """Should propagate NetworkError from complex initialization scenarios"""
        with patch("gemini_batch.batch_processor.GeminiClient") as mock_client_class:
            mock_client_class.side_effect = NetworkError(
                "Network connection timeout during client setup"
            )

            with pytest.raises(
                NetworkError, match="Network connection timeout during client setup"
            ):
                BatchProcessor(
                    api_key="test_key_123456789012345678901234567890",
                    model_name="custom-model",
                    enable_caching=True,
                    additional_param="test_value",  # Extra kwargs
                )

            # Verify all parameters were passed correctly before the error occurred
            mock_client_class.assert_called_once_with(
                api_key="test_key_123456789012345678901234567890",
                model_name="custom-model",
                enable_caching=True,
                additional_param="test_value",
            )

    def test_error_handling_does_not_interfere_with_provided_client(
        self, mock_genai_client
    ):
        """Should not trigger error handling paths when using pre-configured client"""
        from gemini_batch.client import GeminiClient

        # Create a valid client instance
        mock_client = GeminiClient(api_key="test_key_123456789012345678901234567890")

        # This should work fine and not hit any error handling paths
        processor = BatchProcessor(client=mock_client)

        assert processor.client is mock_client
        # Verify reset_metrics was called (indicating successful initialization)
        assert hasattr(processor, "individual_calls")
        assert processor.individual_calls == 0


class TestBatchProcessorValidation:
    """Test input validation and error cases"""

    def test_rejects_empty_questions_list(self, batch_processor):
        """Should raise ValueError when questions list is empty"""
        with pytest.raises(ValueError, match="Content and questions are required"):
            batch_processor.process_text_questions("Some content", [])

    def test_rejects_empty_content(self, batch_processor):
        """Should raise ValueError when content is empty"""
        with pytest.raises(ValueError, match="Content and questions are required"):
            batch_processor.process_text_questions("", ["Question?"])

    def test_rejects_none_content(self, batch_processor):
        """Should raise ValueError when content is None"""
        with pytest.raises(ValueError, match="Content and questions are required"):
            batch_processor.process_text_questions(None, ["Question?"])


class TestBatchProcessorCoreFunctionality:
    """Test core processing functionality"""

    def test_processes_single_question_correctly(
        self, batch_processor, mock_genai_client
    ):
        """Should handle single question and return properly structured result"""
        # Use a response that extract_answers can actually parse
        realistic_response = type(SAMPLE_RESPONSES["simple_answer"])(
            text="Answer 1: This is a comprehensive answer to the question.",
            usage_metadata=SAMPLE_RESPONSES["simple_answer"].usage_metadata,
        )
        mock_genai_client.models.generate_content.return_value = realistic_response

        result = batch_processor.process_text_questions("Content", ["Single question?"])

        assert len(result["batch_answers"]) == 1
        assert "This is a comprehensive answer" in result["batch_answers"][0]
        assert result["question_count"] == 1
        assert result["metrics"]["batch"]["calls"] == 1

    def test_processes_multiple_questions_in_batch(
        self, batch_processor, mock_genai_client
    ):
        """Should process multiple questions in a single batch call"""
        questions = [f"Question {i}?" for i in range(1, 4)]  # 3 questions

        # Create realistic batch response
        realistic_response = type(SAMPLE_RESPONSES["batch_answer"])(
            text="""Answer 1: First comprehensive answer.
                   Answer 2: Second detailed response.
                   Answer 3: Third thorough explanation.""",
            usage_metadata=SAMPLE_RESPONSES["batch_answer"].usage_metadata,
        )
        mock_genai_client.models.generate_content.return_value = realistic_response

        result = batch_processor.process_text_questions("Content", questions)

        assert len(result["batch_answers"]) == 3
        assert result["question_count"] == 3
        assert result["metrics"]["batch"]["calls"] == 1
        # Verify each answer has substantive content
        for answer in result["batch_answers"]:
            assert len(answer.strip()) > 10

    def test_comparison_mode_disabled_by_default(
        self, batch_processor, mock_genai_client
    ):
        """Should not perform individual processing unless explicitly requested"""
        mock_genai_client.models.generate_content.return_value = SAMPLE_RESPONSES[
            "batch_answer"
        ]

        result = batch_processor.process_text_questions(
            "Content", ["Q1?", "Q2?"], compare_methods=False
        )

        assert "individual_answers" not in result
        assert result["efficiency"]["comparison_available"] is False
        # Should only make 1 API call (batch)
        assert mock_genai_client.models.generate_content.call_count == 1

    def test_metrics_tracking_works(self, batch_processor, mock_genai_client):
        """Should track metrics for batch processing"""
        mock_genai_client.models.generate_content.return_value = SAMPLE_RESPONSES[
            "simple_answer"
        ]

        result = batch_processor.process_text_questions("Content", ["Q1?", "Q2?"])

        assert "efficiency" in result
        assert "metrics" in result
        assert "batch" in result["metrics"]
        assert result["metrics"]["batch"]["calls"] == 1
        assert "prompt_tokens" in result["metrics"]["batch"]


class TestBatchProcessorErrorHandling:
    """Test error handling and fallback behavior"""

    def test_batch_failure_triggers_individual_fallback(
        self, batch_processor, mock_genai_client
    ):
        """When batch processing fails, should fallback to individual processing"""
        # First call (batch) fails with retries, subsequent calls (individual) succeed
        # Client retries up to 3 times total (1 + 2 retries) for batch, then individual calls
        mock_genai_client.models.generate_content.side_effect = [
            NetworkError("Batch processing failed"),  # Initial batch attempt
            NetworkError("Batch processing failed"),  # Retry 1
            NetworkError("Batch processing failed"),  # Retry 2 (final)
        ] + [SAMPLE_RESPONSES["simple_answer"]] * 2  # Individual fallback calls

        result = batch_processor.process_text_questions("Content", ["Q1?", "Q2?"])

        # Should make 5 calls: 3 failed batch attempts + 2 successful individual calls
        assert mock_genai_client.models.generate_content.call_count == 5
        assert len(result["batch_answers"]) == 2
        # Metrics should reflect fallback to individual processing
        assert result["metrics"]["batch"]["calls"] == 2  # Fallback individual calls

    def test_graceful_fallback_on_batch_failure(
        self, batch_processor, mock_genai_client
    ):
        """Should fall back to individual processing when batch fails"""
        # Setup: batch fails, individual succeeds
        mock_genai_client.models.generate_content.side_effect = [
            # Batch attempt with retries (3 total failures)
            NetworkError("Batch failed"),
            NetworkError("Batch failed"),
            NetworkError("Batch failed"),
            # Individual fallback calls succeed
            SAMPLE_RESPONSES["simple_answer"],
            SAMPLE_RESPONSES["simple_answer"],
        ]

        questions = ["Q1?", "Q2?"]
        result = batch_processor.process_text_questions("Content", questions)

        # Should have answers despite batch failure
        assert len(result["batch_answers"]) == len(questions)
        assert all("Error:" not in answer for answer in result["batch_answers"])

        # Should reflect fallback in metrics
        assert result["metrics"]["batch"]["calls"] == len(questions)  # Individual calls

    def test_partial_failure_handling(self, batch_processor, mock_genai_client):
        """Should handle scenarios where some operations fail"""
        # Setup: batch fails, first individual succeeds, second fails
        mock_genai_client.models.generate_content.side_effect = [
            NetworkError("Batch failed"),
            NetworkError("Retry 1"),
            NetworkError("Retry 2"),
            SAMPLE_RESPONSES["simple_answer"],  # First individual succeeds
            APIError("Individual failed"),  # Second individual fails
        ]

        result = batch_processor.process_text_questions("Content", ["Q1?", "Q2?"])

        # Should have both successful and error responses
        assert len(result["batch_answers"]) == 2
        assert "Error:" not in result["batch_answers"][0]  # Success
        assert "Error:" in result["batch_answers"][1]  # Failure

    def test_complete_failure_returns_error_messages(
        self, batch_processor, mock_genai_client
    ):
        """When both batch and individual processing fail, should return error messages"""
        mock_genai_client.models.generate_content.side_effect = APIError(
            "Complete failure"
        )

        result = batch_processor.process_text_questions("Content", ["Question?"])

        assert len(result["batch_answers"]) == 1
        assert "Error:" in result["batch_answers"][0]
        assert "Complete failure" in result["batch_answers"][0]

    def test_preserves_partial_results_on_mixed_failures(
        self, batch_processor, mock_genai_client
    ):
        """Should preserve successful results when some individual calls fail"""
        # Batch fails with retries, then mixed individual results
        mock_genai_client.models.generate_content.side_effect = [
            NetworkError("Batch failed"),  # Initial batch attempt
            NetworkError("Batch failed"),  # Retry 1
            NetworkError("Batch failed"),  # Retry 2 (final)
            SAMPLE_RESPONSES["simple_answer"],  # Individual success
            APIError("Individual failed"),  # Individual failure
        ]

        result = batch_processor.process_text_questions("Content", ["Q1?", "Q2?"])

        assert len(result["batch_answers"]) == 2
        # First should have real content, second should have error message
        assert "Error:" not in result["batch_answers"][0]
        assert "Error:" in result["batch_answers"][1]
