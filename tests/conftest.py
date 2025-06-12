from unittest.mock import Mock, patch

import pytest

from gemini_batch import BatchProcessor, GeminiClient

from .fixtures.api_responses import (
    MockResponse,
    MockUsageMetadata,
)


@pytest.fixture
def mock_api_key():
    """Provide test API key"""
    return "test_api_key_12345"


@pytest.fixture
def mock_env(mock_api_key, monkeypatch):
    """Mock environment variables"""
    monkeypatch.setenv("GEMINI_API_KEY", mock_api_key)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")


@pytest.fixture
def mock_genai_client():
    """Mock the underlying genai.Client"""
    with patch("gemini_batch.client.genai.Client") as mock:
        client_instance = Mock()
        mock.return_value = client_instance
        yield client_instance


@pytest.fixture
def gemini_client(mock_env, mock_genai_client):
    """Provide configured GeminiClient for testing"""
    return GeminiClient()


@pytest.fixture
def batch_processor(gemini_client):
    """Provide BatchProcessor for testing"""
    return BatchProcessor()


@pytest.fixture
def sample_content():
    """Reusable test content"""
    return """
    Artificial Intelligence represents one of the most transformative 
    technologies of the 21st century, enabling machines to process 
    natural language and solve complex problems.
    """


@pytest.fixture
def sample_questions():
    """Reusable test questions"""
    return [
        "What makes AI transformative?",
        "How does AI process language?",
        "What problems can AI solve?",
    ]


# Response builder for flexible test scenarios
@pytest.fixture
def response_builder():
    """Factory for creating custom mock responses"""

    def _build_response(text: str, prompt_tokens: int = 100, output_tokens: int = 50):
        return MockResponse(
            text=text,
            usage_metadata=MockUsageMetadata(
                prompt_token_count=prompt_tokens, candidates_token_count=output_tokens
            ),
        )

    return _build_response
