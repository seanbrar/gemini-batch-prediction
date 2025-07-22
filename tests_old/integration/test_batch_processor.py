import json
from unittest.mock import MagicMock

import pytest

from tests.fixtures.api_responses import SAMPLE_RESPONSES


@pytest.mark.integration
class TestBatchProcessorIntegration:
    """Test BatchProcessor with mocked API calls"""

    def test_process_text_questions_comparison_mode(
        self, batch_processor, sample_content, sample_questions, mock_genai_client
    ):
        """Test full workflow with comparison - verifies both batch and individual processing work together"""
        # Setup realistic batch response that extract_answers can actually parse
        batch_response_text = json.dumps(
            [
                "AI is transformative because it enables machines to process natural language.",
                "AI processes language through neural networks and pattern recognition.",
                "AI can solve complex problems like medical diagnosis and financial analysis.",
            ]
        )

        # Mock responses with proper structure
        individual_response = SAMPLE_RESPONSES["simple_answer"]
        batch_response = type(individual_response)(
            text=batch_response_text, usage_metadata=individual_response.usage_metadata
        )

        # Set up the call sequence: batch call first, then individual calls for comparison
        # The batch call will trigger one API call, then individual calls will trigger 3 more
        batch_processor.client.generate_batch = MagicMock(return_value=batch_response)
        batch_processor.client.generate_content = MagicMock(
            return_value=individual_response
        )

        results = batch_processor.process_questions(
            sample_content, sample_questions, compare_methods=True
        )

        # Verify structure
        assert "efficiency" in results
        assert "answers" in results
        assert "individual_answers" in results
        assert "metrics" in results

        # Verify comparison was actually performed
        assert results["efficiency"]["comparison_available"] is True
        assert len(results["answers"]) == len(sample_questions)
        assert len(results["individual_answers"]) == len(sample_questions)

        # Verify API was called the expected number of times (1 batch + 3 individual)
        assert batch_processor.client.generate_batch.call_count == 1
        assert batch_processor.client.generate_content.call_count == len(
            sample_questions
        )

        # Verify metrics contain real data
        assert results["metrics"]["batch"]["calls"] == 1
        assert results["metrics"]["individual"]["calls"] == len(sample_questions)

    def test_batch_fallback_on_error(
        self, batch_processor, sample_content, sample_questions, mock_genai_client
    ):
        """Test fallback to individual processing when batch fails"""
        # Batch call fails with retries, subsequent calls (individual) succeed
        # Client retries up to 3 times total (1 + 2 retries) for batch
        batch_processor.client.generate_batch = MagicMock(
            side_effect=Exception("Batch processing failed")
        )
        batch_processor.client.generate_content = MagicMock(
            return_value=SAMPLE_RESPONSES["simple_answer"]
        )

        results = batch_processor.process_questions(sample_content, sample_questions)

        # Verify fallback occurred by checking the call pattern
        assert batch_processor.client.generate_batch.call_count == 1
        assert batch_processor.client.generate_content.call_count == len(
            sample_questions
        )

        # Verify results structure
        assert "answers" in results
        assert len(results["answers"]) == len(sample_questions)

        # Verify metrics show the fallback behavior
        assert results["metrics"]["batch"]["calls"] == 0
        assert results["metrics"]["individual"]["calls"] == len(sample_questions)
        assert (
            results["efficiency"]["comparison_available"] is False
        )  # No comparison since batch failed

    def test_end_to_end_workflow_realistic_scenario(
        self, batch_processor, mock_genai_client
    ):
        """Test a realistic end-to-end scenario with actual content processing"""
        content = "Machine learning is a subset of AI that enables computers to learn patterns from data."
        questions = [
            "What is machine learning?",
            "How does it relate to AI?",
            "What does it enable computers to do?",
        ]

        # Realistic batch response in JSON format
        batch_response_text = json.dumps(
            [
                "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                "Machine learning is part of the broader field of AI, providing the learning capabilities.",
                "It enables computers to identify patterns and make predictions from data without explicit programming.",
            ]
        )

        batch_response = type(SAMPLE_RESPONSES["simple_answer"])(
            text=batch_response_text,
            usage_metadata=SAMPLE_RESPONSES["simple_answer"].usage_metadata,
        )

        batch_processor.client.generate_batch = MagicMock(return_value=batch_response)

        results = batch_processor.process_questions(content, questions)

        # Verify all answers were extracted properly
        assert len(results["answers"]) == 3
        for i, answer in enumerate(results["answers"], 1):
            assert len(answer.strip()) > 10  # Should have substantial content

        # Verify efficiency tracking
        assert "efficiency" in results
        assert "metrics" in results
        assert results["metrics"]["batch"]["calls"] == 1
        assert batch_processor.client.generate_batch.call_count == 1
