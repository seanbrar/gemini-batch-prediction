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
        # Setup cache manager mock
        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            # Setup cache strategy to enable caching
            mock_cache_strategy = Mock()
            mock_cache_strategy.should_cache = True
            mock_cache_strategy.strategy_type = "explicit"
            mock_cache_manager.analyze_cache_strategy.return_value = mock_cache_strategy

            # Setup cache result
            mock_cache_result = Mock()
            mock_cache_result.success = True
            mock_cache_result.cache_name = "test_cache_123"
            mock_cache_result.cached_tokens = 500
            mock_cache_manager.attempt_cache_retrieval.return_value = mock_cache_result

            # Use content that exceeds the cache analysis threshold
            from gemini_batch.constants import MIN_TEXT_LENGTH_FOR_CACHE_ANALYSIS

            long_content = EDUCATIONAL_CONTENT["academic_paper_excerpt"] * (
                int(
                    MIN_TEXT_LENGTH_FOR_CACHE_ANALYSIS
                    / len(EDUCATIONAL_CONTENT["academic_paper_excerpt"])
                )
                + 2
            )
            questions = ["What is the main research question?"]

            # Process content
            result = batch_processor.process_text_questions(
                content=long_content, questions=questions
            )

            # Verify cache manager was called
            mock_cache_manager.analyze_cache_strategy.assert_called_once()
            mock_cache_manager.attempt_cache_retrieval.assert_called_once()

            # Verify result structure
            assert result["success"] is True
            assert "cache_summary" in result
            assert result["cache_summary"]["cache_hit"] is True

    def test_cache_hit_uses_cache_reference(self, batch_processor, mock_genai_client):
        """Test that cache hits use cache references."""
        # Setup cache manager mock
        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            # Setup cache strategy to enable caching
            mock_cache_strategy = Mock()
            mock_cache_strategy.should_cache = True
            mock_cache_strategy.strategy_type = "explicit"
            mock_cache_manager.analyze_cache_strategy.return_value = mock_cache_strategy

            # Setup cache result with cache hit
            mock_cache_result = Mock()
            mock_cache_result.success = True
            mock_cache_result.cache_name = "test_cache_456"
            mock_cache_result.cached_tokens = 300
            mock_cache_manager.attempt_cache_retrieval.return_value = mock_cache_result

            # Use content that exceeds the cache analysis threshold
            from gemini_batch.constants import MIN_TEXT_LENGTH_FOR_CACHE_ANALYSIS

            long_content = EDUCATIONAL_CONTENT["academic_paper_excerpt"] * (
                int(
                    MIN_TEXT_LENGTH_FOR_CACHE_ANALYSIS
                    / len(EDUCATIONAL_CONTENT["academic_paper_excerpt"])
                )
                + 2
            )
            questions = ["What are the key findings?"]

            # Process content
            result = batch_processor.process_text_questions(
                content=long_content, questions=questions
            )

            # Verify cache manager was called
            mock_cache_manager.analyze_cache_strategy.assert_called_once()
            mock_cache_manager.attempt_cache_retrieval.assert_called_once()

            # Verify result structure
            assert result["success"] is True
            assert "cache_summary" in result
            assert result["cache_summary"]["cache_hit"] is True
            assert "test_cache_456" in str(result["cache_summary"])

    def test_cache_miss_triggers_api_call(self, batch_processor, mock_genai_client):
        """Test that cache misses trigger API calls."""
        # Setup cache manager mock
        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            # Setup cache strategy to enable caching
            mock_cache_strategy = Mock()
            mock_cache_strategy.should_cache = True
            mock_cache_strategy.strategy_type = "explicit"
            mock_cache_manager.analyze_cache_strategy.return_value = mock_cache_strategy

            # Setup cache result with cache miss
            mock_cache_result = Mock()
            mock_cache_result.success = False
            mock_cache_result.cache_name = None
            mock_cache_result.cached_tokens = 0
            mock_cache_manager.attempt_cache_retrieval.return_value = mock_cache_result

            # Setup API response
            response = Mock()
            response.text = "Answer from API call"
            response.usage_metadata = Mock()
            response.usage_metadata.prompt_token_count = 150
            response.usage_metadata.candidates_token_count = 75
            response.usage_metadata.cached_content_token_count = 0

            mock_genai_client.models.generate_content.return_value = response

            # Use content that exceeds the cache analysis threshold
            from gemini_batch.constants import MIN_TEXT_LENGTH_FOR_CACHE_ANALYSIS

            long_content = EDUCATIONAL_CONTENT["academic_paper_excerpt"] * (
                int(
                    MIN_TEXT_LENGTH_FOR_CACHE_ANALYSIS
                    / len(EDUCATIONAL_CONTENT["academic_paper_excerpt"])
                )
                + 2
            )
            questions = ["What is the methodology?"]

            # Process content
            result = batch_processor.process_text_questions(
                content=long_content, questions=questions
            )

            # Verify cache manager was called
            mock_cache_manager.analyze_cache_strategy.assert_called_once()
            mock_cache_manager.attempt_cache_retrieval.assert_called_once()

            # Verify API was called
            mock_genai_client.models.generate_content.assert_called_once()

            # Verify result structure
            assert result["success"] is True
            assert "cache_summary" in result
            assert result["cache_summary"]["cache_hit"] is False

    def test_cache_failure_falls_back_gracefully(
        self, batch_processor, mock_genai_client
    ):
        """Test that cache failures fall back to API calls gracefully."""
        # Setup cache manager mock
        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            # Setup cache strategy to enable caching
            mock_cache_strategy = Mock()
            mock_cache_strategy.should_cache = True
            mock_cache_strategy.strategy_type = "explicit"
            mock_cache_manager.analyze_cache_strategy.return_value = mock_cache_strategy

            # Setup cache result with failure
            mock_cache_result = Mock()
            mock_cache_result.success = False
            mock_cache_result.cache_name = None
            mock_cache_result.cached_tokens = 0
            mock_cache_manager.attempt_cache_retrieval.side_effect = Exception(
                "Cache error"
            )

            # Setup API response
            response = Mock()
            response.text = "Answer from API fallback"
            response.usage_metadata = Mock()
            response.usage_metadata.prompt_token_count = 200
            response.usage_metadata.candidates_token_count = 100
            response.usage_metadata.cached_content_token_count = 0

            mock_genai_client.models.generate_content.return_value = response

            # Use content that exceeds the cache analysis threshold
            from gemini_batch.constants import MIN_TEXT_LENGTH_FOR_CACHE_ANALYSIS

            long_content = EDUCATIONAL_CONTENT["academic_paper_excerpt"] * (
                int(
                    MIN_TEXT_LENGTH_FOR_CACHE_ANALYSIS
                    / len(EDUCATIONAL_CONTENT["academic_paper_excerpt"])
                )
                + 2
            )
            questions = ["What are the implications?"]

            # Process content
            result = batch_processor.process_text_questions(
                content=long_content, questions=questions
            )

            # Verify cache manager was called
            mock_cache_manager.analyze_cache_strategy.assert_called_once()
            mock_cache_manager.attempt_cache_retrieval.assert_called_once()

            # Verify API was called as fallback
            mock_genai_client.models.generate_content.assert_called_once()

            # Verify result structure
            assert result["success"] is True
            assert "cache_summary" in result
            assert result["cache_summary"]["cache_hit"] is False

    def test_caching_with_different_questions_on_same_content(
        self, batch_processor, mock_genai_client
    ):
        """Test that caching works with different questions on the same content."""
        # Setup cache manager mock
        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            # Setup cache strategy to enable caching
            mock_cache_strategy = Mock()
            mock_cache_strategy.should_cache = True
            mock_cache_strategy.strategy_type = "explicit"
            mock_cache_manager.analyze_cache_strategy.return_value = mock_cache_strategy

            # Setup cache result with cache hit
            mock_cache_result = Mock()
            mock_cache_result.success = True
            mock_cache_result.cache_name = "test_cache_789"
            mock_cache_result.cached_tokens = 400
            mock_cache_manager.attempt_cache_retrieval.return_value = mock_cache_result

            # Use content that exceeds the cache analysis threshold
            from gemini_batch.constants import MIN_TEXT_LENGTH_FOR_CACHE_ANALYSIS

            long_content = EDUCATIONAL_CONTENT["academic_paper_excerpt"] * (
                int(
                    MIN_TEXT_LENGTH_FOR_CACHE_ANALYSIS
                    / len(EDUCATIONAL_CONTENT["academic_paper_excerpt"])
                )
                + 2
            )
            questions = ["What is the conclusion?", "What are the limitations?"]

            # Process content
            result = batch_processor.process_text_questions(
                content=long_content, questions=questions
            )

            # Verify cache manager was called
            mock_cache_manager.analyze_cache_strategy.assert_called_once()
            mock_cache_manager.attempt_cache_retrieval.assert_called_once()

            # Verify result structure
            assert result["success"] is True
            assert "cache_summary" in result
            assert result["cache_summary"]["cache_hit"] is True
            assert len(result["answers"]) == 2

    def test_explicit_vs_implicit_caching_strategies(
        self, batch_processor, mock_genai_client
    ):
        """Test explicit vs implicit caching strategies."""
        # Setup cache manager mock
        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            # Setup cache strategy for explicit caching
            mock_cache_strategy = Mock()
            mock_cache_strategy.should_cache = True
            mock_cache_strategy.strategy_type = "explicit"
            mock_cache_manager.analyze_cache_strategy.return_value = mock_cache_strategy

            # Setup cache result
            mock_cache_result = Mock()
            mock_cache_result.success = True
            mock_cache_result.cache_name = "explicit_cache"
            mock_cache_manager.get_or_create_cache.return_value = mock_cache_result

            # Setup response
            response = Mock()
            response.text = "Answer with explicit caching"
            response.usage_metadata = Mock()
            response.usage_metadata.prompt_token_count = 100
            response.usage_metadata.candidates_token_count = 50
            response.usage_metadata.cached_content_token_count = 400

            mock_genai_client.models.generate_content.return_value = response

            content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
            questions = ["What is the main research question?"]

            # Test explicit caching
            result_explicit = batch_processor.process_questions(
                content, questions, enable_caching=True
            )

            # Verify explicit caching was used
            assert len(result_explicit["answers"]) == 1
            assert result_explicit["cache_summary"]["cached_tokens"] == 400

    def test_caching_disabled_behavior(self, batch_processor, mock_genai_client):
        """Test behavior when caching is disabled."""
        # Setup response without caching
        response = Mock()
        response.text = "Answer without caching"
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 50
        response.usage_metadata.cached_content_token_count = 0

        mock_genai_client.models.generate_content.return_value = response

        content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
        questions = ["What is the main research question?"]

        result = batch_processor.process_questions(
            content, questions, enable_caching=False
        )

        # Verify no caching was attempted
        assert len(result["answers"]) == 1
        assert len(result["answers"][0]) > 0

        # Verify metrics show no caching
        assert result["cache_summary"]["cache_enabled"] is False

    def test_caching_with_multimodal_content(self, batch_processor, mock_genai_client):
        """Test caching with multimodal content."""
        # Setup cache manager mock
        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            # Setup cache strategy to enable caching
            mock_cache_strategy = Mock()
            mock_cache_strategy.should_cache = True
            mock_cache_strategy.strategy_type = "explicit"
            mock_cache_manager.analyze_cache_strategy.return_value = mock_cache_strategy

            # Setup cache result
            mock_cache_result = Mock()
            mock_cache_result.success = True
            mock_cache_result.cache_name = "multimodal_cache"
            mock_cache_manager.get_or_create_cache.return_value = mock_cache_result

            # Setup response
            response = Mock()
            response.text = "Answer with multimodal content"
            response.usage_metadata = Mock()
            response.usage_metadata.prompt_token_count = 150
            response.usage_metadata.candidates_token_count = 75
            response.usage_metadata.cached_content_token_count = 300

            mock_genai_client.models.generate_content.return_value = response

            # Use multimodal content (image + text)
            content = [
                EDUCATIONAL_CONTENT["short_lesson"],
                "test_files/congratulations-9607355_1280.png",  # Image file
            ]
            questions = ["What is shown in the image and how does it relate to AI?"]

            result = batch_processor.process_questions(
                content, questions, enable_caching=True
            )

            # Verify caching was attempted
            mock_cache_manager.get_or_create_cache.assert_called_once()

            # Verify result
            assert len(result["answers"]) == 1
            assert len(result["answers"][0]) > 0

    def test_error_handling_in_caching(self, batch_processor, mock_genai_client):
        """Test error handling in caching workflows."""
        # Setup cache manager mock to raise exception
        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            mock_cache_manager.analyze_cache_strategy.side_effect = Exception(
                "Cache manager error"
            )

            # Setup response for fallback processing
            response = Mock()
            response.text = "Answer despite cache error"
            response.usage_metadata = Mock()
            response.usage_metadata.prompt_token_count = 100
            response.usage_metadata.candidates_token_count = 50
            response.usage_metadata.cached_content_token_count = 0

            mock_genai_client.models.generate_content.return_value = response

            content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
            questions = ["What is the main research question?"]

            result = batch_processor.process_questions(
                content, questions, enable_caching=True
            )

            # Verify fallback processing worked despite cache error
            assert len(result["answers"]) == 1
            assert len(result["answers"][0]) > 0

    def test_caching_performance_comparison(self, batch_processor, mock_genai_client):
        """Test performance comparison between cached and non-cached processing."""
        # Setup cache manager mock
        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            # Setup cache strategy to enable caching
            mock_cache_strategy = Mock()
            mock_cache_strategy.should_cache = True
            mock_cache_strategy.strategy_type = "explicit"
            mock_cache_manager.analyze_cache_strategy.return_value = mock_cache_strategy

            # Setup cache result
            mock_cache_result = Mock()
            mock_cache_result.success = True
            mock_cache_result.cache_name = "performance_cache"
            mock_cache_manager.get_or_create_cache.return_value = mock_cache_result

            # Setup response with caching
            cached_response = Mock()
            cached_response.text = "Answer with caching"
            cached_response.usage_metadata = Mock()
            cached_response.usage_metadata.prompt_token_count = 100
            cached_response.usage_metadata.candidates_token_count = 50
            cached_response.usage_metadata.cached_content_token_count = 400

            mock_genai_client.models.generate_content.return_value = cached_response

            content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
            questions = ["What is the main research question?"]

            result_cached = batch_processor.process_questions(
                content, questions, enable_caching=True
            )

            # Verify caching performance metrics
            assert len(result_cached["answers"]) == 1
            cached_tokens = result_cached["cache_summary"]["cached_tokens"]
            assert cached_tokens == 400

    def test_multiple_requests_caching(self, batch_processor, mock_genai_client):
        """Test caching behavior across multiple requests."""
        # Setup cache manager mock
        with patch.object(
            batch_processor.client, "cache_manager"
        ) as mock_cache_manager:
            # Setup cache strategy to enable caching
            mock_cache_strategy = Mock()
            mock_cache_strategy.should_cache = True
            mock_cache_strategy.strategy_type = "explicit"
            mock_cache_manager.analyze_cache_strategy.return_value = mock_cache_strategy

            # Setup cache results for multiple requests
            mock_cache_results = []
            for i in range(3):
                mock_result = Mock()
                mock_result.success = True
                mock_result.cache_name = f"cache_request_{i}"
                mock_cache_results.append(mock_result)

            mock_cache_manager.get_or_create_cache.side_effect = mock_cache_results

            # Setup responses
            responses = []
            for i in range(3):
                response = Mock()
                response.text = f"Answer for request {i}"
                response.usage_metadata = Mock()
                response.usage_metadata.prompt_token_count = 100
                response.usage_metadata.candidates_token_count = 50
                response.usage_metadata.cached_content_token_count = 200
                responses.append(response)

            mock_genai_client.models.generate_content.side_effect = responses

            content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
            questions = [
                "What is the main research question?",
                "What methodology was used?",
                "What are the key findings?",
            ]

            result = batch_processor.process_questions(
                content, questions, enable_caching=True
            )

            # Verify all requests used caching
            assert mock_cache_manager.get_or_create_cache.call_count == 3

            # Verify results
            assert len(result["answers"]) == 3
            for answer in result["answers"]:
                assert len(answer) > 0
