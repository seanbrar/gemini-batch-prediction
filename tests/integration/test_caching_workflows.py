"""
Integration tests for caching workflows.
"""

from unittest.mock import Mock, patch

from tests.fixtures.content_samples import EDUCATIONAL_CONTENT


class TestCachingWorkflows:
    """Test caching workflow scenarios."""

    def test_caching_is_attempted_for_large_content(
        self, batch_processor, mock_genai_client
    ):
        """Test that caching is attempted for large content."""
        # Setup large content that should trigger caching
        large_content = "x" * 10000  # Large content

        # Mock cache manager to return successful cache result
        mock_cache_result = Mock()
        mock_cache_result.is_success = True
        mock_cache_result.cache_reference = "cache_ref_123"

        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            mock_cache_manager.get_or_create_cache.return_value = mock_cache_result

            # Setup response for cached generation
            cached_response = Mock()
            cached_response.text = "Answer using cached content"
            cached_response.usage_metadata = Mock()
            cached_response.usage_metadata.prompt_token_count = 100
            cached_response.usage_metadata.candidates_token_count = 50
            cached_response.usage_metadata.cached_content_token_count = 500

            mock_genai_client.generate_batch.return_value = cached_response

            questions = ["What is the main topic?"]

            result = batch_processor.process_questions(
                large_content, questions, enable_caching=True
            )

            # Verify cache manager was called
            mock_cache_manager.get_or_create_cache.assert_called_once()

            # Verify batch generation was called with cache reference
            mock_genai_client.generate_batch.assert_called_once()

            # Verify caching metrics
            assert "cache" in result["metrics"]
            cache_metrics = result["metrics"]["cache"]
            assert cache_metrics["cached_tokens"] == 500
            assert cache_metrics["total_tokens"] > 0

    def test_cache_hit_uses_cache_reference(self, batch_processor, mock_genai_client):
        """Test that cache hits use cache reference instead of full content."""
        # Setup cache hit
        mock_cache_result = Mock()
        mock_cache_result.is_success = True
        mock_cache_result.cache_reference = "cache_ref_456"

        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            mock_cache_manager.get_or_create_cache.return_value = mock_cache_result

            # Setup response for cached generation
            cached_response = Mock()
            cached_response.text = "Answer using cache reference"
            cached_response.usage_metadata = Mock()
            cached_response.usage_metadata.prompt_token_count = 50
            cached_response.usage_metadata.candidates_token_count = 25
            cached_response.usage_metadata.cached_content_token_count = 800

            mock_genai_client.generate_batch.return_value = cached_response

            content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
            questions = ["What is the research question?"]

            result = batch_processor.process_questions(
                content, questions, enable_caching=True
            )

            # Verify cache manager was called
            mock_cache_manager.get_or_create_cache.assert_called_once()

            # Verify batch generation was called
            mock_genai_client.generate_batch.assert_called_once()

            # Verify high cached token count indicates cache was used
            assert result["metrics"]["cache"]["cached_tokens"] == 800
            assert result["metrics"]["cache"]["total_tokens"] > 0

    def test_cache_failure_falls_back_gracefully(
        self, batch_processor, mock_genai_client
    ):
        """Test that cache failures fall back to non-cached generation."""
        # Setup cache failure
        mock_cache_result = Mock()
        mock_cache_result.is_success = False
        mock_cache_result.error = "Cache creation failed"

        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            mock_cache_manager.get_or_create_cache.return_value = mock_cache_result

            # Setup response for non-cached generation
            fallback_response = Mock()
            fallback_response.text = "Answer without caching"
            fallback_response.usage_metadata = Mock()
            fallback_response.usage_metadata.prompt_token_count = 200
            fallback_response.usage_metadata.candidates_token_count = 100
            fallback_response.usage_metadata.cached_content_token_count = 0

            mock_genai_client.generate_batch.return_value = fallback_response

            content = EDUCATIONAL_CONTENT["medium_article"]
            questions = ["What is the main topic?"]

            result = batch_processor.process_questions(
                content, questions, enable_caching=True
            )

            # Verify cache manager was called
            mock_cache_manager.get_or_create_cache.assert_called_once()

            # Verify batch generation was called (fallback)
            mock_genai_client.generate_batch.assert_called_once()

            # Verify no cached tokens (fallback used)
            assert result["metrics"]["cache"]["cached_tokens"] == 0
            assert result["metrics"]["cache"]["total_tokens"] > 0

    def test_caching_with_different_questions_on_same_content(
        self, batch_processor, mock_genai_client
    ):
        """Test caching with different questions on the same content."""
        # Setup cache manager to return same cache for same content
        mock_cache_result = Mock()
        mock_cache_result.is_success = True
        mock_cache_result.cache_reference = "cache_ref_789"

        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            mock_cache_manager.get_or_create_cache.return_value = mock_cache_result

            # Setup responses for different questions
            response1 = Mock()
            response1.text = "Answer to first question"
            response1.usage_metadata = Mock()
            response1.usage_metadata.prompt_token_count = 100
            response1.usage_metadata.candidates_token_count = 50
            response1.usage_metadata.cached_content_token_count = 300

            response2 = Mock()
            response2.text = "Answer to second question"
            response2.usage_metadata = Mock()
            response2.usage_metadata.prompt_token_count = 80
            response2.usage_metadata.candidates_token_count = 40
            response2.usage_metadata.cached_content_token_count = 300

            mock_genai_client.generate_batch.side_effect = [response1, response2]

            content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]

            # First question
            result1 = batch_processor.process_questions(
                content, ["What is the research question?"], enable_caching=True
            )

            # Second question on same content
            result2 = batch_processor.process_questions(
                content, ["What methodology was used?"], enable_caching=True
            )

            # Verify cache manager was called twice (same content)
            assert mock_cache_manager.get_or_create_cache.call_count == 2

            # Verify both used caching
            assert result1["metrics"]["cache"]["cached_tokens"] == 300
            assert result2["metrics"]["cache"]["cached_tokens"] == 300

    def test_explicit_vs_implicit_caching_strategies(
        self, batch_processor, mock_genai_client
    ):
        """Test explicit vs implicit caching strategies."""
        # Setup cache manager
        mock_cache_result = Mock()
        mock_cache_result.is_success = True
        mock_cache_result.cache_reference = "cache_ref_explicit"

        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            mock_cache_manager.get_or_create_cache.return_value = mock_cache_result

            # Setup response
            response = Mock()
            response.text = "Cached answer"
            response.usage_metadata = Mock()
            response.usage_metadata.prompt_token_count = 100
            response.usage_metadata.candidates_token_count = 50
            response.usage_metadata.cached_content_token_count = 400

            mock_genai_client.generate_batch.return_value = response

            content = EDUCATIONAL_CONTENT["medium_article"]
            questions = ["What is the main topic?"]

            # Test explicit caching
            result_explicit = batch_processor.process_questions(
                content, questions, enable_caching=True, cache_strategy="explicit"
            )

            # Test implicit caching
            result_implicit = batch_processor.process_questions(
                content, questions, enable_caching=True, cache_strategy="implicit"
            )

            # Verify both used caching
            assert result_explicit["metrics"]["cache"]["cached_tokens"] == 400
            assert result_implicit["metrics"]["cache"]["cached_tokens"] == 400

            # Verify cache manager was called for both strategies
            assert mock_cache_manager.get_or_create_cache.call_count == 2

    def test_caching_disabled_behavior(self, batch_processor, mock_genai_client):
        """Test behavior when caching is disabled."""
        # Setup response without caching
        response = Mock()
        response.text = "Answer without caching"
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 200
        response.usage_metadata.candidates_token_count = 100
        response.usage_metadata.cached_content_token_count = 0

        mock_genai_client.generate_batch.return_value = response

        content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
        questions = ["What is the research question?"]

        # Test with caching disabled
        result = batch_processor.process_questions(
            content, questions, enable_caching=False
        )

        # Verify batch generation was called
        mock_genai_client.generate_batch.assert_called_once()

        # Verify no cached tokens
        assert result["metrics"]["cache"]["cached_tokens"] == 0
        assert result["metrics"]["cache"]["total_tokens"] > 0

    def test_caching_with_multimodal_content(self, batch_processor, mock_genai_client):
        """Test caching with multimodal content."""
        # Setup cache manager
        mock_cache_result = Mock()
        mock_cache_result.is_success = True
        mock_cache_result.cache_reference = "cache_ref_multimodal"

        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            mock_cache_manager.get_or_create_cache.return_value = mock_cache_result

            # Setup response for multimodal content
            response = Mock()
            response.text = "Answer about multimodal content"
            response.usage_metadata = Mock()
            response.usage_metadata.prompt_token_count = 150
            response.usage_metadata.candidates_token_count = 75
            response.usage_metadata.cached_content_token_count = 600

            mock_genai_client.generate_batch.return_value = response

            # Create multimodal content (text + file reference)
            multimodal_content = [
                EDUCATIONAL_CONTENT["short_lesson"],
                "/path/to/document.pdf",
            ]

            questions = ["What are the key themes across these sources?"]

            result = batch_processor.process_questions_multi_source(
                multimodal_content, questions, enable_caching=True
            )

            # Verify cache manager was called
            mock_cache_manager.get_or_create_cache.assert_called_once()

            # Verify batch generation was called
            mock_genai_client.generate_batch.assert_called_once()

            # Verify caching was used
            assert result["metrics"]["cache"]["cached_tokens"] == 600
            assert result["metrics"]["cache"]["total_tokens"] > 0

    def test_error_handling_in_caching(self, batch_processor, mock_genai_client):
        """Test error handling in caching workflows."""
        # Setup cache manager to raise exception
        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            mock_cache_manager.get_or_create_cache.side_effect = Exception(
                "Cache error"
            )

            # Setup fallback response
            fallback_response = Mock()
            fallback_response.text = "Fallback answer"
            fallback_response.usage_metadata = Mock()
            fallback_response.usage_metadata.prompt_token_count = 200
            fallback_response.usage_metadata.candidates_token_count = 100
            fallback_response.usage_metadata.cached_content_token_count = 0

            mock_genai_client.generate_batch.return_value = fallback_response

            content = EDUCATIONAL_CONTENT["medium_article"]
            questions = ["What is the main topic?"]

            # Should handle cache error gracefully
            result = batch_processor.process_questions(
                content, questions, enable_caching=True
            )

            # Verify cache manager was called
            mock_cache_manager.get_or_create_cache.assert_called_once()

            # Verify fallback was used
            mock_genai_client.generate_batch.assert_called_once()

            # Verify no cached tokens (fallback)
            assert result["metrics"]["cache"]["cached_tokens"] == 0
            assert result["metrics"]["cache"]["total_tokens"] > 0

    def test_caching_performance_comparison(self, batch_processor, mock_genai_client):
        """Test performance comparison between cached and non-cached requests."""
        # Setup cache manager
        mock_cache_result = Mock()
        mock_cache_result.is_success = True
        mock_cache_result.cache_reference = "cache_ref_perf"

        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            mock_cache_manager.get_or_create_cache.return_value = mock_cache_result

            # Setup responses with different token counts
            cached_response = Mock()
            cached_response.text = "Cached answer"
            cached_response.usage_metadata = Mock()
            cached_response.usage_metadata.prompt_token_count = 50
            cached_response.usage_metadata.candidates_token_count = 25
            cached_response.usage_metadata.cached_content_token_count = 800

            non_cached_response = Mock()
            non_cached_response.text = "Non-cached answer"
            non_cached_response.usage_metadata = Mock()
            non_cached_response.usage_metadata.prompt_token_count = 850
            non_cached_response.usage_metadata.candidates_token_count = 25
            non_cached_response.usage_metadata.cached_content_token_count = 0

            mock_genai_client.generate_batch.side_effect = [
                cached_response,
                non_cached_response,
            ]

            content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
            questions = ["What is the research question?"]

            # Test with caching
            result_cached = batch_processor.process_questions(
                content, questions, enable_caching=True
            )

            # Test without caching
            result_non_cached = batch_processor.process_questions(
                content, questions, enable_caching=False
            )

            # Verify performance difference
            cached_tokens = result_cached["metrics"]["cache"]["total_tokens"]
            non_cached_tokens = result_non_cached["metrics"]["cache"]["total_tokens"]

            # Cached should use fewer effective tokens
            assert cached_tokens < non_cached_tokens

            # Verify caching was used in first case
            assert result_cached["metrics"]["cache"]["cached_tokens"] == 800
            assert result_non_cached["metrics"]["cache"]["cached_tokens"] == 0

    def test_multiple_requests_caching(self, batch_processor, mock_genai_client):
        """Test caching across multiple requests."""
        # Setup cache manager
        mock_cache_result = Mock()
        mock_cache_result.is_success = True
        mock_cache_result.cache_reference = "cache_ref_multi"

        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            mock_cache_manager.get_or_create_cache.return_value = mock_cache_result

            # Setup responses for multiple requests
            responses = [
                Mock(
                    text=f"Answer {i}",
                    usage_metadata=Mock(
                        prompt_token_count=100,
                        candidates_token_count=50,
                        cached_content_token_count=400,
                    ),
                )
                for i in range(3)
            ]

            mock_genai_client.generate_batch.side_effect = responses

            content = EDUCATIONAL_CONTENT["medium_article"]

            # Make multiple requests with same content
            results = []
            for i in range(3):
                result = batch_processor.process_questions(
                    content, [f"Question {i}?"], enable_caching=True
                )
                results.append(result)

            # Verify cache manager was called for each request
            assert mock_cache_manager.get_or_create_cache.call_count == 3

            # Verify all used caching
            for result in results:
                assert result["metrics"]["cache"]["cached_tokens"] == 400
                assert result["metrics"]["cache"]["total_tokens"] > 0
