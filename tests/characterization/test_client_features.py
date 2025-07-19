"""
Characterization tests for advanced features of the GeminiClient,
such as caching and multimodal handling.
"""

from unittest.mock import MagicMock

import pytest


@pytest.mark.golden_test("golden_files/test_client_caching.yml")
def test_explicit_caching_behavior(
    golden, caching_gemini_client, mocked_internal_genai_client
):
    """
    Characterizes the explicit caching flow within the GeminiClient.

    This test simulates two calls with the same large content. It verifies
    that the first call creates a cache and the second call uses it.
    """
    _ = golden["input"]  # Silence unused field warning
    # Arrange
    large_content = (
        "A very large piece of text content..." * 1000
    )  # Exceeds caching threshold
    question = "What is the summary?"
    interaction_log = []  # This will be our golden output

    # We define a side effect to log the interactions with the mocked API.
    # This lets us capture the behavior of our client.
    def log_generate_content(*args, **kwargs):  # noqa: ARG001
        call_details = {"method": "generate_content"}
        # Check if the call is using a cache
        if (
            hasattr(kwargs.get("config"), "cached_content")
            and kwargs.get("config").cached_content
        ):
            call_details["cached_content"] = kwargs["config"].cached_content
        else:
            call_details["cached_content"] = None
        interaction_log.append(call_details)

        # Return a realistic mock response object with the necessary attributes
        mock_response = MagicMock()
        mock_response.text = "Mocked response"
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.cached_content_token_count = 0
        return mock_response

    def log_cache_create(*args, **kwargs):  # noqa: ARG001
        interaction_log.append({"method": "caches.create"})
        # The conftest fixture already configures the return value for this
        return mocked_internal_genai_client.caches.create.return_value

    # Attach the logging side effects to our mock
    mocked_internal_genai_client.models.generate_content.side_effect = (
        log_generate_content
    )
    mocked_internal_genai_client.caches.create.side_effect = log_cache_create

    # Act
    # First call: should trigger a cache miss and create a cache.
    caching_gemini_client.generate_content(large_content, prompt=question)

    # Second call: should trigger a cache hit and use the existing cache.
    caching_gemini_client.generate_content(large_content, prompt=question)

    # Assert
    # We assert that the recorded sequence of interactions matches our golden file.
    assert interaction_log == golden.out["interaction_log"]
