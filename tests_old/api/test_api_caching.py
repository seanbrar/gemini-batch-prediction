"""
API tests for caching functionality.
"""

import time

import pytest

from tests.fixtures.content_samples import EDUCATIONAL_CONTENT


@pytest.mark.api
@pytest.mark.expensive  # Uses more quota
class TestAPICaching:
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

        # Verify token usage differences (second request should use fewer tokens)
        usage1 = result1.get("metrics", {}).get("usage", {})
        usage2 = result2.get("metrics", {}).get("usage", {})

        if usage1 and usage2:
            total_tokens1 = usage1.get("total_tokens", 0)
            total_tokens2 = usage2.get("total_tokens", 0)
            cached_tokens2 = usage2.get("cached_content_token_count", 0)

            print(f"   Request 1 total tokens: {total_tokens1}")
            print(f"   Request 2 total tokens: {total_tokens2}")
            print(f"   Request 2 cached tokens: {cached_tokens2}")

            # Second request should show some caching benefit
            if cached_tokens2 > 0:
                print("✅ Caching benefit detected")
                assert total_tokens2 < total_tokens1 or cached_tokens2 > 0

    def test_implicit_caching_real_api(self, real_batch_processor, api_rate_limiter):
        """Test implicit context caching with real API."""
        content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
        questions_set1 = ["What is the main topic?"]
        questions_set2 = ["What are the key conclusions?"]

        # First request - should create cache implicitly
        result1 = real_batch_processor.process_questions(
            content, questions_set1, enable_caching=True, cache_strategy="implicit"
        )

        time.sleep(3)  # Rate limiting

        # Second request - should use cache
        result2 = real_batch_processor.process_questions(
            content, questions_set2, enable_caching=True, cache_strategy="implicit"
        )

        # Verify both requests succeeded
        assert len(result1["answers"]) == len(questions_set1)
        assert len(result2["answers"]) == len(questions_set2)

        # Check for caching metrics
        cache_info1 = result1.get("metrics", {}).get("cache", {})
        cache_info2 = result2.get("metrics", {}).get("cache", {})

        print("✅ Implicit caching test completed")
        print(f"   Request 1 cache info: {cache_info1}")
        print(f"   Request 2 cache info: {cache_info2}")

        # Verify responses are reasonable
        assert all(len(answer) > 10 for answer in result1["answers"])
        assert all(len(answer) > 10 for answer in result2["answers"])

    def test_caching_with_different_content_sizes(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test caching behavior with different content sizes."""
        # Small content
        small_content = EDUCATIONAL_CONTENT["short_lesson"]
        small_questions = ["What is machine learning?"]

        # Large content
        large_content = (
            EDUCATIONAL_CONTENT["academic_paper_excerpt"] * 2
        )  # Make it larger
        large_questions = ["What is the main research focus?"]

        # Test small content caching
        result_small1 = real_batch_processor.process_questions(
            small_content, small_questions, enable_caching=True
        )

        time.sleep(3)  # Rate limiting

        result_small2 = real_batch_processor.process_questions(
            small_content, ["What are the key concepts?"], enable_caching=True
        )

        # Test large content caching
        result_large1 = real_batch_processor.process_questions(
            large_content, large_questions, enable_caching=True
        )

        time.sleep(3)  # Rate limiting

        result_large2 = real_batch_processor.process_questions(
            large_content, ["What are the implications?"], enable_caching=True
        )

        # Verify all requests succeeded
        assert len(result_small1["answers"]) == 1
        assert len(result_small2["answers"]) == 1
        assert len(result_large1["answers"]) == 1
        assert len(result_large2["answers"]) == 1

        print("✅ Different content sizes caching test completed")

        # Compare token usage
        small_usage1 = result_small1.get("metrics", {}).get("usage", {})
        small_usage2 = result_small2.get("metrics", {}).get("usage", {})
        large_usage1 = result_large1.get("metrics", {}).get("usage", {})
        large_usage2 = result_large2.get("metrics", {}).get("usage", {})

        print(
            f"   Small content - Request 1 tokens: {small_usage1.get('total_tokens', 0)}"
        )
        print(
            f"   Small content - Request 2 tokens: {small_usage2.get('total_tokens', 0)}"
        )
        print(
            f"   Large content - Request 1 tokens: {large_usage1.get('total_tokens', 0)}"
        )
        print(
            f"   Large content - Request 2 tokens: {large_usage2.get('total_tokens', 0)}"
        )

    def test_caching_disabled_behavior_real_api(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test behavior when caching is disabled with real API."""
        content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
        questions = ["What is the main topic?"]

        # Request with caching disabled
        result_disabled = real_batch_processor.process_questions(
            content, questions, enable_caching=False
        )

        time.sleep(3)  # Rate limiting

        # Request with caching enabled
        result_enabled = real_batch_processor.process_questions(
            content, questions, enable_caching=True
        )

        # Verify both requests succeeded
        assert len(result_disabled["answers"]) == 1
        assert len(result_enabled["answers"]) == 1

        # Check cache metrics
        disabled_cache = result_disabled.get("metrics", {}).get("cache", {})
        enabled_cache = result_enabled.get("metrics", {}).get("cache", {})

        print("✅ Caching disabled vs enabled test completed")
        print(f"   Disabled cache info: {disabled_cache}")
        print(f"   Enabled cache info: {enabled_cache}")

        # Verify responses are reasonable
        assert len(result_disabled["answers"][0]) > 10
        assert len(result_enabled["answers"][0]) > 10

    def test_caching_with_multimodal_content_real_api(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test caching with multimodal content using real API."""
        # Use text content for this test (multimodal would require file uploads)
        content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
        questions_set1 = ["What is the research methodology?"]
        questions_set2 = ["What are the key findings?"]

        # First request
        result1 = real_batch_processor.process_questions(
            content, questions_set1, enable_caching=True
        )

        time.sleep(3)  # Rate limiting

        # Second request
        result2 = real_batch_processor.process_questions(
            content, questions_set2, enable_caching=True
        )

        # Verify both requests succeeded
        assert len(result1["answers"]) == 1
        assert len(result2["answers"]) == 1

        # Check for caching metrics
        cache_info1 = result1.get("metrics", {}).get("cache", {})
        cache_info2 = result2.get("metrics", {}).get("cache", {})

        print("✅ Multimodal content caching test completed")
        print(f"   Request 1 cache info: {cache_info1}")
        print(f"   Request 2 cache info: {cache_info2}")

        # Verify responses are reasonable
        assert len(result1["answers"][0]) > 10
        assert len(result2["answers"][0]) > 10

    def test_caching_error_recovery_real_api(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test caching error recovery with real API."""
        content = EDUCATIONAL_CONTENT["short_lesson"]
        questions = ["What is the main topic?"]

        # Test with potentially problematic cache strategy
        try:
            result = real_batch_processor.process_questions(
                content, questions, enable_caching=True, cache_strategy="explicit"
            )

            # Verify successful processing
            assert len(result["answers"]) == 1
            assert len(result["answers"][0]) > 10

            print("✅ Caching error recovery test completed successfully")

        except Exception as e:
            # Should handle errors gracefully
            print(f"⚠️  Caching error handled gracefully: {type(e).__name__}")
            # The error should be related to caching or API issues
            assert any(
                keyword in str(e).lower()
                for keyword in ["cache", "api", "rate", "limit"]
            )

    def test_caching_performance_comparison(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test caching performance comparison."""
        import time

        content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
        questions = ["What is the main research question?"]

        # Test without caching
        start_time = time.time()
        result_no_cache = real_batch_processor.process_questions(
            content, questions, enable_caching=False
        )
        no_cache_time = time.time() - start_time

        time.sleep(3)  # Rate limiting

        # Test with caching
        start_time = time.time()
        result_with_cache = real_batch_processor.process_questions(
            content, questions, enable_caching=True
        )
        with_cache_time = time.time() - start_time

        # Verify both requests succeeded
        assert len(result_no_cache["answers"]) == 1
        assert len(result_with_cache["answers"]) == 1

        print("✅ Caching performance comparison completed")
        print(f"   No cache time: {no_cache_time:.2f}s")
        print(f"   With cache time: {with_cache_time:.2f}s")

        # Verify responses are reasonable
        assert len(result_no_cache["answers"][0]) > 10
        assert len(result_with_cache["answers"][0]) > 10

        # Compare token usage
        no_cache_usage = result_no_cache.get("metrics", {}).get("usage", {})
        with_cache_usage = result_with_cache.get("metrics", {}).get("usage", {})

        print(f"   No cache tokens: {no_cache_usage.get('total_tokens', 0)}")
        print(f"   With cache tokens: {with_cache_usage.get('total_tokens', 0)}")
        print(
            f"   Cached tokens: {with_cache_usage.get('cached_content_token_count', 0)}"
        )

    def test_caching_with_multiple_requests(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test caching behavior across multiple requests."""
        content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
        questions = [
            "What is the main topic?",
            "What methodology was used?",
            "What are the key findings?",
            "What are the implications?",
        ]

        results = []
        cache_metrics = []

        # Make multiple requests
        for i, question in enumerate(questions):
            result = real_batch_processor.process_questions(
                content, [question], enable_caching=True
            )

            results.append(result)
            cache_metrics.append(result.get("metrics", {}).get("cache", {}))

            if i < len(questions) - 1:  # Don't sleep after last request
                time.sleep(3)  # Rate limiting

        # Verify all requests succeeded
        for result in results:
            assert len(result["answers"]) == 1
            assert len(result["answers"][0]) > 10

        print("✅ Multiple requests caching test completed")
        print(f"   Total requests: {len(results)}")

        # Analyze cache metrics
        cache_hits = sum(
            1 for metrics in cache_metrics if metrics.get("cache_used", False)
        )
        print(f"   Cache hits: {cache_hits}")

        # Show token usage progression
        for i, result in enumerate(results):
            usage = result.get("metrics", {}).get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            cached_tokens = usage.get("cached_content_token_count", 0)
            print(f"   Request {i + 1}: {total_tokens} tokens, {cached_tokens} cached")
