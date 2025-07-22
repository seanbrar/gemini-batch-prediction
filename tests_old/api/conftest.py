"""
Configuration for real API integration tests.
"""

import os

import pytest

from gemini_batch import BatchProcessor, GeminiClient


@pytest.fixture(scope="session")
def real_api_client():
    """Real Gemini client for API integration tests."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY required for API tests")

    return GeminiClient(
        api_key=api_key,
        model="gemini-2.0-flash",
        tier="tier_1",  # Explicitly use tier 1 in order to test caching
    )


@pytest.fixture(scope="session")
def real_batch_processor(real_api_client):
    """Real batch processor for API integration tests."""
    return BatchProcessor(client=real_api_client)


@pytest.fixture
def api_rate_limiter():
    """Ensure API tests don't exceed rate limits."""
    import time

    # Add delay between tests to stay well under limit
    time.sleep(5)  # 5 second delay between API tests
    yield
    time.sleep(1)  # Additional delay after test


@pytest.fixture
def small_test_content():
    """Small content to minimize token usage in API tests."""
    return {
        "text": "Artificial intelligence enables machines to learn and make decisions. Machine learning is a subset of AI that uses data to improve performance.",
        "questions": [
            "What is AI?",
            "How does ML relate to AI?",
            "What enables machine learning?",
        ],
    }
