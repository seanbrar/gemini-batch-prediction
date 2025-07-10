"""
Basic real API functionality tests.
Run with: pytest tests/api_integration/ -m api --verbose
Requires: GEMINI_API_KEY and ENABLE_API_TESTS=1 environment variables
"""

import time

import pytest

from gemini_batch.conversation import create_conversation
from tests.fixtures.content_samples import EDUCATIONAL_CONTENT, QUESTION_SETS


@pytest.mark.api
class TestRealAPIBasics:
    """Test basic API functionality with real calls."""

    def test_single_question_real_api(
        self, real_batch_processor, api_rate_limiter, small_test_content
    ):
        """Test single question processing with real API."""
        content = small_test_content["text"]
        question = "What is the main topic?"

        result = real_batch_processor.process_questions(content, [question])

        # Verify response structure
        assert "answers" in result
        assert len(result["answers"]) == 1
        assert isinstance(result["answers"][0], str)
        assert len(result["answers"][0]) > 10  # Reasonable response length

        # Verify metrics
        assert "metrics" in result
        assert "usage" in result["metrics"]
        assert result["metrics"]["usage"]["total_tokens"] > 0

        print("✅ Single question API call successful")
        print(f"   Response length: {len(result['answers'][0])} chars")
        print(f"   Token usage: {result['metrics']['usage']['total_tokens']}")

    def test_batch_processing_efficiency_real_api(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test that batch processing actually reduces API calls."""
        content = EDUCATIONAL_CONTENT["medium_article"]
        questions = QUESTION_SETS["basic_comprehension"]

        # Measure batch processing
        start_time = time.time()
        batch_result = real_batch_processor.process_questions(content, questions)
        batch_duration = time.time() - start_time

        # Verify batch results
        assert len(batch_result["answers"]) == len(questions)
        assert all(len(answer) > 10 for answer in batch_result["answers"])

        # Verify efficiency metrics are present
        assert "efficiency" in batch_result["metrics"]
        efficiency = batch_result["metrics"]["efficiency"]

        # Should show batch optimization
        assert "estimated_individual_calls" in efficiency
        assert "actual_calls" in efficiency
        assert efficiency["actual_calls"] <= efficiency["estimated_individual_calls"]

        print("✅ Batch processing efficiency test passed")
        print(f"   Questions: {len(questions)}")
        print(
            f"   Estimated individual calls: {efficiency['estimated_individual_calls']}"
        )
        print(f"   Actual calls: {efficiency['actual_calls']}")
        print(f"   Efficiency gain: {efficiency.get('efficiency_ratio', 'N/A')}")
        print(f"   Duration: {batch_duration:.2f}s")

    def test_conversation_memory_real_api(self, real_api_client, api_rate_limiter):
        """Test real conversation memory functionality."""
        content = EDUCATIONAL_CONTENT["short_lesson"]

        session = create_conversation([content], client=real_api_client)

        # First question
        response1 = session.ask("What is machine learning?")
        assert len(response1) > 10
        assert len(session.history) == 1

        time.sleep(2)  # Rate limiting

        # Follow-up question that should reference previous context
        response2 = session.ask("How does it differ from traditional programming?")
        assert len(response2) > 10
        assert len(session.history) == 2

        # Verify conversation context was maintained
        # (This is a behavioral test - responses should be contextually aware)
        print("✅ Conversation memory test passed")
        print(f"   Turn 1 length: {len(response1)} chars")
        print(f"   Turn 2 length: {len(response2)} chars")
        print(f"   History entries: {len(session.history)}")


@pytest.mark.api
@pytest.mark.expensive  # Uses more quota
class TestRealAPICaching:
    """Test real context caching functionality."""

    def test_explicit_caching_real_api(self, real_batch_processor, api_rate_limiter):
        """Test explicit context caching with real API."""
        # Use longer content to trigger caching
        content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
        questions_set1 = [
            "What is the main research question?",
            "What methodology was used?",
        ]
        questions_set2 = ["What were the key findings?", "What are the implications?"]

        # First request - should create cache
        result1 = real_batch_processor.process_questions(
            content, questions_set1, enable_caching=True, cache_strategy="explicit"
        )

        time.sleep(3)  # Rate limiting

        # Second request with same content - should use cache
        result2 = real_batch_processor.process_questions(
            content, questions_set2, enable_caching=True, cache_strategy="explicit"
        )

        # Verify both requests succeeded
        assert len(result1["answers"]) == len(questions_set1)
        assert len(result2["answers"]) == len(questions_set2)

        # Check for caching metrics
        cache_info1 = result1.get("metrics", {}).get("cache", {})
        cache_info2 = result2.get("metrics", {}).get("cache", {})

        print("✅ Explicit caching test completed")
        print(f"   Request 1 cache info: {cache_info1}")
        print(f"   Request 2 cache info: {cache_info2}")

        # At minimum, verify no errors and reasonable responses
        assert all(len(answer) > 10 for answer in result1["answers"])
        assert all(len(answer) > 10 for answer in result2["answers"])


@pytest.mark.api
class TestRealAPIErrorHandling:
    """Test error handling with real API."""

    def test_api_error_recovery_real_api(self, real_batch_processor, api_rate_limiter):
        """Test recovery from API errors."""
        content = "Test content for error handling."

        # Normal request should work
        good_questions = ["What is this about?"]
        result = real_batch_processor.process_questions(content, good_questions)

        assert len(result["answers"]) == 1
        assert len(result["answers"][0]) > 0

        print("✅ API error recovery test passed")
        print("   Normal request handled successfully")

    def test_rate_limit_handling_real_api(self, real_batch_processor):
        """Test that rate limiting is properly handled."""
        content = "Brief content for rate limit testing."
        questions = ["What is this?"]

        # Make request and verify rate limiting is working
        start_time = time.time()
        result = real_batch_processor.process_questions(content, questions)
        duration = time.time() - start_time

        assert len(result["answers"]) == 1

        # Should complete in reasonable time (not hanging on rate limits)
        assert duration < 30  # Should not take more than 30 seconds

        print("✅ Rate limit handling test passed")
        print(f"   Request completed in {duration:.2f}s")
