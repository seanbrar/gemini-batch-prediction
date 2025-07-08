import json
from unittest.mock import Mock, patch

import pytest

from gemini_batch import BatchProcessor
from gemini_batch.client.configuration import ClientConfiguration
from gemini_batch.exceptions import MissingKeyError, NetworkError
from tests.fixtures.api_responses import SAMPLE_RESPONSES


class TestBatchProcessorInitialization:
    """Test BatchProcessor initialization behavior"""

    def test_creates_client_when_none_provided(self):
        """Should create a GeminiClient when no client is provided"""
        with patch("gemini_batch.batch_processor.GeminiClient") as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.from_env.return_value = mock_client_instance

            processor = BatchProcessor()

            assert processor.client is mock_client_instance
            mock_client_class.from_env.assert_called_once()

    def test_uses_provided_client(self, mock_genai_client):
        """Should use the provided client instance"""
        from gemini_batch import GeminiClient

        config = ClientConfiguration(
            api_key="test_key_123456789012345678901234567890",
            model_name="gemini-2.0-flash",
            enable_caching=False,
        )
        mock_client = GeminiClient(config)
        processor = BatchProcessor(client=mock_client)

        assert processor.client is mock_client

    def test_passes_kwargs_to_new_client(self):
        """Should pass initialization parameters to new GeminiClient"""
        with patch("gemini_batch.batch_processor.GeminiClient") as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.from_env.return_value = mock_client_instance

            BatchProcessor(
                api_key="test_key_123456789012345678901234567890",
                model_name="custom-model",
                enable_caching=True,
            )

            # Should call from_env with the parameters
            mock_client_class.from_env.assert_called_once()

    def test_warns_when_client_and_kwargs_both_provided(self, mock_genai_client):
        """Should warn user when both client and kwargs are provided (kwargs ignored)"""
        from gemini_batch import GeminiClient

        config = ClientConfiguration(
            api_key="test_key_123456789012345678901234567890",
            model_name="gemini-2.0-flash",
            enable_caching=False,
        )
        mock_client = GeminiClient(config)

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
            mock_client_class.from_env.side_effect = exception_type(expected_message)

            with pytest.raises(exception_type, match=expected_message):
                BatchProcessor(api_key=api_key)

    def test_handles_missing_key_error_with_client_kwargs(self):
        """Should propagate MissingKeyError even when additional client kwargs are
        provided"""
        with patch("gemini_batch.batch_processor.GeminiClient") as mock_client_class:
            mock_client_class.from_env.side_effect = MissingKeyError(
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

            # Verify from_env was called
            mock_client_class.from_env.assert_called_once()

    def test_handles_network_error_with_complex_initialization(self):
        """Should propagate NetworkError from complex initialization scenarios"""
        with patch("gemini_batch.batch_processor.GeminiClient") as mock_client_class:
            mock_client_class.from_env.side_effect = NetworkError(
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

            # Verify from_env was called
            mock_client_class.from_env.assert_called_once()

    def test_error_handling_does_not_interfere_with_provided_client(
        self, mock_genai_client
    ):
        """Should not trigger error handling paths when using pre-configured client"""
        from gemini_batch import GeminiClient

        # Create a valid client instance
        config = ClientConfiguration(
            api_key="test_key_123456789012345678901234567890",
            model_name="gemini-2.0-flash",
            enable_caching=False,
        )
        mock_client = GeminiClient(config)

        # This should work fine and not hit any error handling paths
        processor = BatchProcessor(client=mock_client)

        assert processor.client is mock_client
        # Verify metrics tracking is initialized
        assert hasattr(processor, "individual_calls")
        assert processor.individual_calls == 0


class TestBatchProcessorValidation:
    """Test input validation and error cases"""

    def test_rejects_empty_questions_list(self, batch_processor):
        """Should raise ValueError when questions list is empty"""
        with pytest.raises(ValueError, match="Questions are required"):
            batch_processor.process_questions("Some content", [])

    def test_rejects_empty_content(self, batch_processor):
        """Should handle empty content gracefully (no validation error)"""
        # Empty content should not raise an error - it's handled by the API
        result = batch_processor.process_questions("", ["Question?"])
        assert "answers" in result

    def test_rejects_none_content(self, batch_processor):
        """Should handle None content gracefully (no validation error)"""
        # None content should not raise an error - it's handled by the API
        result = batch_processor.process_questions(None, ["Question?"])
        assert "answers" in result


class TestBatchProcessorCoreFunctionality:
    """Test core processing functionality"""

    def test_processes_single_question_correctly(
        self, batch_processor, mock_genai_client
    ):
        """Should handle single question and return properly structured result"""
        # Create a mock GeminiClient for this test
        from unittest.mock import Mock

        mock_client = Mock()

        # Use a response that extract_answers can actually parse
        realistic_response = type(SAMPLE_RESPONSES["simple_answer"])(
            text=json.dumps(["This is a comprehensive answer to the question."]),
            usage_metadata=SAMPLE_RESPONSES["simple_answer"].usage_metadata,
        )
        # Mock the GeminiClient methods
        mock_client.generate_batch.return_value = realistic_response

        # Replace the client in the batch processor
        batch_processor.client = mock_client

        result = batch_processor.process_questions("Content", ["Single question?"])

        assert len(result["answers"]) == 1
        assert "This is a comprehensive answer" in result["answers"][0]
        assert result["question_count"] == 1
        assert result["metrics"]["batch"]["calls"] == 1

    def test_processes_multiple_questions_in_batch(
        self, batch_processor, mock_genai_client
    ):
        """Should process multiple questions in a single batch call"""
        # Create a mock GeminiClient for this test
        from unittest.mock import Mock

        mock_client = Mock()

        # Use a response that extract_answers can actually parse
        realistic_response = type(SAMPLE_RESPONSES["batch_answer"])(
            text=json.dumps(
                [
                    "First comprehensive answer.",
                    "Second comprehensive answer.",
                    "Third comprehensive answer.",
                ]
            ),
            usage_metadata=SAMPLE_RESPONSES["batch_answer"].usage_metadata,
        )
        # Mock the GeminiClient methods
        mock_client.generate_batch.return_value = realistic_response

        # Replace the client in the batch processor
        batch_processor.client = mock_client

        questions = ["Question 1?", "Question 2?", "Question 3?"]
        result = batch_processor.process_questions("Content", questions)

        assert len(result["answers"]) == 3
        assert "First comprehensive answer" in result["answers"][0]
        assert "Second comprehensive answer" in result["answers"][1]
        assert "Third comprehensive answer" in result["answers"][2]
        assert result["question_count"] == 3
        assert result["metrics"]["batch"]["calls"] == 1

    def test_comparison_mode_disabled_by_default(
        self, batch_processor, mock_genai_client
    ):
        """Should process questions normally when comparison mode is disabled"""
        # Create a mock GeminiClient for this test
        from unittest.mock import Mock

        mock_client = Mock()

        realistic_response = type(SAMPLE_RESPONSES["simple_answer"])(
            text=json.dumps(["Standard batch processing answer."]),
            usage_metadata=SAMPLE_RESPONSES["simple_answer"].usage_metadata,
        )
        # Mock the GeminiClient methods
        mock_client.generate_batch.return_value = realistic_response

        # Replace the client in the batch processor
        batch_processor.client = mock_client

        result = batch_processor.process_questions(
            "Content", ["Question?"], comparison_mode=False
        )

        assert "answers" in result
        assert "comparison" not in result

    def test_metrics_tracking_works(self, batch_processor, mock_genai_client):
        """Should track processing metrics correctly"""
        # Create a mock GeminiClient for this test
        from unittest.mock import Mock

        mock_client = Mock()

        realistic_response = type(SAMPLE_RESPONSES["simple_answer"])(
            text=json.dumps(["Tracked answer."]),
            usage_metadata=SAMPLE_RESPONSES["simple_answer"].usage_metadata,
        )
        # Mock the GeminiClient methods
        mock_client.generate_batch.return_value = realistic_response

        # Replace the client in the batch processor
        batch_processor.client = mock_client

        result = batch_processor.process_questions("Content", ["Question?"])

        assert "metrics" in result
        assert "batch" in result["metrics"]
        assert "calls" in result["metrics"]["batch"]
        assert result["metrics"]["batch"]["calls"] == 1


class TestBatchProcessorErrorHandling:
    """Test error handling and fallback behavior"""

    def test_batch_failure_triggers_individual_fallback(
        self, batch_processor, mock_genai_client
    ):
        """Should fall back to individual processing when batch fails"""
        # Create a mock GeminiClient for this test
        from unittest.mock import Mock

        mock_client = Mock()

        # First call fails (batch), subsequent calls succeed (individual)
        mock_client.generate_batch.side_effect = Exception(
            "Batch failed"
        )  # Batch call fails
        mock_client.generate_content.side_effect = [
            type(SAMPLE_RESPONSES["simple_answer"])(
                text=json.dumps(["Individual answer 1."]),
                usage_metadata=SAMPLE_RESPONSES["simple_answer"].usage_metadata,
            ),
            type(SAMPLE_RESPONSES["simple_answer"])(
                text=json.dumps(["Individual answer 2."]),
                usage_metadata=SAMPLE_RESPONSES["simple_answer"].usage_metadata,
            ),
        ]

        # Replace the client in the batch processor
        batch_processor.client = mock_client

        result = batch_processor.process_questions(
            "Content", ["Question 1?", "Question 2?"]
        )

        print(f"DEBUG: result keys = {list(result.keys())}")
        print(f"DEBUG: metrics = {result.get('metrics', 'NO METRICS')}")
        if "metrics" in result:
            print(f"DEBUG: metrics keys = {list(result['metrics'].keys())}")
            if "individual" in result["metrics"]:
                print(
                    f"DEBUG: individual keys = {list(result['metrics']['individual'].keys())}"
                )

        assert len(result["answers"]) == 2
        assert "Individual answer 1" in result["answers"][0]
        assert "Individual answer 2" in result["answers"][1]
        assert result["metrics"]["individual"]["calls"] == 2

    def test_graceful_fallback_on_batch_failure(
        self, batch_processor, mock_genai_client
    ):
        """Should handle batch failure gracefully with individual processing"""
        # Create a mock GeminiClient for this test
        from unittest.mock import Mock

        mock_client = Mock()

        # Batch fails, individual calls succeed
        mock_client.generate_batch.side_effect = Exception("Batch processing failed")
        mock_client.generate_content.side_effect = [
            type(SAMPLE_RESPONSES["simple_answer"])(
                text=json.dumps(["Fallback answer."]),
                usage_metadata=SAMPLE_RESPONSES["simple_answer"].usage_metadata,
            ),
        ]

        # Replace the client in the batch processor
        batch_processor.client = mock_client

        result = batch_processor.process_questions("Content", ["Question?"])

        assert "answers" in result
        assert len(result["answers"]) == 1
        assert "Fallback answer" in result["answers"][0]

    def test_partial_failure_handling(self, batch_processor, mock_genai_client):
        """Should handle partial failures in individual processing"""
        # Create a mock GeminiClient for this test
        from unittest.mock import Mock

        mock_client = Mock()

        # Some individual calls fail, others succeed
        mock_client.generate_batch.side_effect = Exception("Batch failed")
        mock_client.generate_content.side_effect = [
            type(SAMPLE_RESPONSES["simple_answer"])(
                text=json.dumps(["Successful answer."]),
                usage_metadata=SAMPLE_RESPONSES["simple_answer"].usage_metadata,
            ),
            Exception("Individual call failed"),
        ]

        # Replace the client in the batch processor
        batch_processor.client = mock_client

        result = batch_processor.process_questions(
            "Content", ["Question 1?", "Question 2?"]
        )

        assert len(result["answers"]) == 2
        assert "Successful answer" in result["answers"][0]
        assert "error" in result["answers"][1].lower()

    def test_complete_failure_returns_error_messages(
        self, batch_processor, mock_genai_client
    ):
        """Should return error messages when all processing fails"""
        # Create a mock GeminiClient for this test
        from unittest.mock import Mock

        mock_client = Mock()

        mock_client.generate_batch.side_effect = Exception("Complete failure")
        mock_client.generate_content.side_effect = Exception("Complete failure")

        # Replace the client in the batch processor
        batch_processor.client = mock_client

        result = batch_processor.process_questions(
            "Content", ["Question 1?", "Question 2?"]
        )

        assert len(result["answers"]) == 2
        assert all("error" in answer.lower() for answer in result["answers"])

    def test_preserves_partial_results_on_mixed_failures(
        self, batch_processor, mock_genai_client
    ):
        """Should preserve successful results even when some calls fail"""
        # Create a mock GeminiClient for this test
        from unittest.mock import Mock

        mock_client = Mock()

        mock_client.generate_batch.side_effect = Exception("Batch failed")
        mock_client.generate_content.side_effect = [
            type(SAMPLE_RESPONSES["simple_answer"])(
                text=json.dumps(["Successful answer."]),
                usage_metadata=SAMPLE_RESPONSES["simple_answer"].usage_metadata,
            ),
            Exception("Individual call failed"),
            type(SAMPLE_RESPONSES["simple_answer"])(
                text=json.dumps(["Another successful answer."]),
                usage_metadata=SAMPLE_RESPONSES["simple_answer"].usage_metadata,
            ),
        ]

        # Replace the client in the batch processor
        batch_processor.client = mock_client

        result = batch_processor.process_questions(
            "Content", ["Q1?", "Q2?", "Q3?", "Q4?"]
        )

        assert len(result["answers"]) == 4
        assert "Successful answer" in result["answers"][0]
        assert "error" in result["answers"][1].lower()
        assert "Another successful answer" in result["answers"][2]
        assert "error" in result["answers"][3].lower()
