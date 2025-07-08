"""
Integration tests for batch processor workflows.
"""

from unittest.mock import Mock, patch

from tests.fixtures.content_samples import EDUCATIONAL_CONTENT


class TestBatchProcessorWorkflows:
    """Test batch processor workflow scenarios."""

    def test_fallback_to_individual_processing_on_batch_failure(
        self, batch_processor, mock_genai_client
    ):
        """Test fallback to individual processing when batch processing fails."""
        # Setup batch processing to fail
        mock_genai_client.generate_batch.side_effect = Exception(
            "Batch processing failed"
        )

        # Setup individual processing to succeed
        individual_response = Mock()
        individual_response.text = "Individual answer"
        individual_response.usage_metadata = Mock()
        individual_response.usage_metadata.prompt_token_count = 100
        individual_response.usage_metadata.candidates_token_count = 50
        individual_response.usage_metadata.cached_content_token_count = 0
        mock_genai_client.generate_content.return_value = individual_response

        content = EDUCATIONAL_CONTENT["short_lesson"]
        questions = ["What is machine learning?", "What is deep learning?"]

        result = batch_processor.process_questions(content, questions)

        # Verify fallback behavior
        assert len(result["answers"]) == 2
        assert all(len(answer) > 0 for answer in result["answers"])

        # Verify batch method was called first
        mock_genai_client.generate_batch.assert_called_once()

        # Verify individual method was called for each question
        assert mock_genai_client.generate_content.call_count == 2

        # Verify metrics show individual processing
        assert result["metrics"]["individual"]["calls"] == 2
        assert result["metrics"]["batch"]["calls"] == 0

    def test_compare_methods_triggers_both_paths(
        self, batch_processor, mock_genai_client
    ):
        """Test that compare_methods triggers both batch and individual processing."""
        # Setup both batch and individual responses
        batch_response = Mock()
        batch_response.text = "Batch answer 1\nBatch answer 2"
        batch_response.usage_metadata = Mock()
        batch_response.usage_metadata.prompt_token_count = 200
        batch_response.usage_metadata.candidates_token_count = 100
        batch_response.usage_metadata.cached_content_token_count = 0

        individual_response = Mock()
        individual_response.text = "Individual answer"
        individual_response.usage_metadata = Mock()
        individual_response.usage_metadata.prompt_token_count = 100
        individual_response.usage_metadata.candidates_token_count = 50
        individual_response.usage_metadata.cached_content_token_count = 0

        mock_genai_client.generate_batch.return_value = batch_response
        mock_genai_client.generate_content.return_value = individual_response

        content = EDUCATIONAL_CONTENT["medium_article"]
        questions = ["What is AI?", "What is ML?"]

        result = batch_processor.process_questions(
            content, questions, compare_methods=True
        )

        # Verify both methods were called
        mock_genai_client.generate_batch.assert_called_once()
        assert mock_genai_client.generate_content.call_count == 2

        # Verify result contains both batch and individual answers
        assert "answers" in result
        assert "individual_answers" in result
        assert len(result["answers"]) == 2
        assert len(result["individual_answers"]) == 2

        # Verify metrics show both methods
        assert result["metrics"]["batch"]["calls"] == 1
        assert result["metrics"]["individual"]["calls"] == 2

    def test_multisource_processing_with_directory(
        self, batch_processor, mock_genai_client, fs
    ):
        """Test multi-source processing with directory path."""
        # Create fake filesystem with test files
        fs.create_file("/test_dir/file1.txt", contents="Content from file 1")
        fs.create_file("/test_dir/file2.txt", contents="Content from file 2")
        fs.create_file("/test_dir/file3.pdf", contents="PDF content")

        # Setup response for multi-source processing
        multi_response = Mock()
        multi_response.text = "Analysis covering multiple sources"
        multi_response.usage_metadata = Mock()
        multi_response.usage_metadata.prompt_token_count = 300
        multi_response.usage_metadata.candidates_token_count = 150
        multi_response.usage_metadata.cached_content_token_count = 0

        mock_genai_client.generate_batch.return_value = multi_response

        sources = [
            EDUCATIONAL_CONTENT["short_lesson"],
            "/test_dir",  # Directory path
            "Additional text content",
        ]

        questions = ["What are the main themes across these sources?"]

        result = batch_processor.process_questions_multi_source(sources, questions)

        # Verify processing succeeded
        assert len(result["answers"]) == 1
        assert len(result["answers"][0]) > 0

        # Verify batch processing was used
        mock_genai_client.generate_batch.assert_called_once()

        # Verify metrics
        assert result["metrics"]["batch"]["calls"] == 1
        assert result["metrics"]["batch"]["prompt_tokens"] > 0

    def test_caching_enabled_processing(self, batch_processor, mock_genai_client):
        """Test processing with caching enabled."""
        # Setup response with caching
        cached_response = Mock()
        cached_response.text = "Answer with cached content"
        cached_response.usage_metadata = Mock()
        cached_response.usage_metadata.prompt_token_count = 100
        cached_response.usage_metadata.candidates_token_count = 50
        cached_response.usage_metadata.cached_content_token_count = 200

        mock_genai_client.generate_batch.return_value = cached_response

        content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
        questions = ["What is the main research question?"]

        result = batch_processor.process_questions(
            content, questions, enable_caching=True
        )

        # Verify caching was used
        assert len(result["answers"]) == 1
        assert len(result["answers"][0]) > 0

        # Verify cache metrics
        assert "cache" in result["metrics"]
        cache_metrics = result["metrics"]["cache"]
        assert cache_metrics["cached_tokens"] == 200
        assert cache_metrics["total_tokens"] > 0

    def test_structured_output_processing(self, batch_processor, mock_genai_client):
        """Test processing with structured output."""
        from pydantic import BaseModel

        class ResearchSummary(BaseModel):
            main_topic: str
            key_findings: list[str]
            methodology: str

        # Setup structured response
        structured_response = Mock()
        structured_response.text = """
        {
            "main_topic": "Machine Learning Applications",
            "key_findings": ["Improved accuracy", "Faster processing"],
            "methodology": "Experimental analysis"
        }
        """
        structured_response.usage_metadata = Mock()
        structured_response.usage_metadata.prompt_token_count = 150
        structured_response.usage_metadata.candidates_token_count = 75
        structured_response.usage_metadata.cached_content_token_count = 0

        mock_genai_client.generate_batch.return_value = structured_response

        content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
        questions = ["Summarize the research"]

        result = batch_processor.process_questions(
            content, questions, response_schema=ResearchSummary
        )

        # Verify structured output
        assert "structured_data" in result
        assert len(result["structured_data"]) == 1
        structured_data = result["structured_data"][0]

        assert isinstance(structured_data, ResearchSummary)
        assert structured_data.main_topic == "Machine Learning Applications"
        assert len(structured_data.key_findings) == 2
        assert structured_data.methodology == "Experimental analysis"

    def test_error_handling_in_batch_processing(
        self, batch_processor, mock_genai_client
    ):
        """Test error handling during batch processing."""
        # Setup both batch and individual processing to fail initially, then succeed
        mock_genai_client.generate_batch.side_effect = [
            Exception("Batch failed"),
            Mock(
                text="Batch answer",
                usage_metadata=Mock(
                    prompt_token_count=100,
                    candidates_token_count=50,
                    cached_content_token_count=0,
                ),
            ),
        ]

        mock_genai_client.generate_content.side_effect = [
            Exception("Individual failed"),
            Mock(
                text="Individual answer",
                usage_metadata=Mock(
                    prompt_token_count=100,
                    candidates_token_count=50,
                    cached_content_token_count=0,
                ),
            ),
        ]

        content = EDUCATIONAL_CONTENT["short_lesson"]
        questions = ["What is AI?"]

        # Should handle errors gracefully
        result = batch_processor.process_questions(content, questions)

        # Verify error handling
        assert "errors" in result
        assert len(result["errors"]) > 0

        # Verify partial results if any
        if "answers" in result:
            assert all(len(answer) > 0 for answer in result["answers"])

    def test_rate_limiting_in_batch_processing(
        self, batch_processor, mock_genai_client
    ):
        """Test rate limiting behavior in batch processing."""
        # Setup rate limiter to enforce delays
        with patch.object(batch_processor.client, "rate_limiter") as mock_rate_limiter:
            mock_rate_limiter.wait_if_needed.return_value = None

            # Setup successful response
            response = Mock()
            response.text = "Rate limited answer"
            response.usage_metadata = Mock()
            response.usage_metadata.prompt_token_count = 100
            response.usage_metadata.candidates_token_count = 50
            response.usage_metadata.cached_content_token_count = 0

            mock_genai_client.generate_batch.return_value = response

            content = EDUCATIONAL_CONTENT["short_lesson"]
            questions = ["What is AI?"]

            result = batch_processor.process_questions(content, questions)

            # Verify rate limiting was called
            mock_rate_limiter.wait_if_needed.assert_called()

            # Verify processing succeeded
            assert len(result["answers"]) == 1
            assert len(result["answers"][0]) > 0

    def test_token_tracking_in_batch_processing(
        self, batch_processor, mock_genai_client
    ):
        """Test token tracking during batch processing."""
        # Setup response with specific token counts
        response = Mock()
        response.text = "Answer with specific token count"
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 1000
        response.usage_metadata.candidates_token_count = 500
        response.usage_metadata.cached_content_token_count = 200

        mock_genai_client.generate_batch.return_value = response

        content = EDUCATIONAL_CONTENT["medium_article"]
        questions = ["What is the main topic?", "What are the key points?"]

        result = batch_processor.process_questions(content, questions)

        # Verify token tracking
        assert "metrics" in result
        metrics = result["metrics"]

        assert "usage" in metrics
        usage = metrics["usage"]
        assert usage["prompt_tokens"] == 1000
        assert usage["output_tokens"] == 500
        assert usage["cached_tokens"] == 200
        assert usage["total_tokens"] == 1300

    def test_efficiency_calculation_in_batch_processing(
        self, batch_processor, mock_genai_client
    ):
        """Test efficiency calculation in batch processing."""
        # Setup response
        response = Mock()
        response.text = "Efficient batch answer"
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 800
        response.usage_metadata.candidates_token_count = 400
        response.usage_metadata.cached_content_token_count = 100

        mock_genai_client.generate_batch.return_value = response

        content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
        questions = ["What is the research question?", "What methodology was used?"]

        result = batch_processor.process_questions(content, questions)

        # Verify efficiency metrics
        assert "efficiency" in result["metrics"]
        efficiency = result["metrics"]["efficiency"]

        assert "estimated_individual_calls" in efficiency
        assert "actual_calls" in efficiency
        assert efficiency["actual_calls"] <= efficiency["estimated_individual_calls"]

        # Verify cost efficiency
        assert "cost_efficiency_ratio" in efficiency
        assert efficiency["cost_efficiency_ratio"] > 0

    def test_multi_source_with_mixed_content_types(
        self, batch_processor, mock_genai_client, fs
    ):
        """Test multi-source processing with mixed content types."""
        # Create test files
        fs.create_file("/test_dir/document.pdf", contents="PDF content")
        fs.create_file("/test_dir/data.txt", contents="Text data")

        # Setup response
        response = Mock()
        response.text = "Analysis of mixed content types"
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 500
        response.usage_metadata.candidates_token_count = 250
        response.usage_metadata.cached_content_token_count = 0

        mock_genai_client.generate_batch.return_value = response

        sources = [
            EDUCATIONAL_CONTENT["short_lesson"],
            "/test_dir",  # Directory with mixed files
            "https://example.com/document.pdf",  # URL
            "Direct text content",
        ]

        questions = ["What are the common themes across all sources?"]

        result = batch_processor.process_questions_multi_source(sources, questions)

        # Verify processing succeeded
        assert len(result["answers"]) == 1
        assert len(result["answers"][0]) > 0

        # Verify batch processing was used
        mock_genai_client.generate_batch.assert_called_once()

        # Verify metrics
        assert result["metrics"]["batch"]["calls"] == 1
        assert result["metrics"]["batch"]["prompt_tokens"] > 0
