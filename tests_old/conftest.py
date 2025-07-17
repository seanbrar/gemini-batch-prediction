"""
Global test configuration with support for different test types.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from gemini_batch import BatchProcessor, GeminiClient
from gemini_batch.client.configuration import ClientConfiguration

from .fixtures.api_responses import (
    MockResponse,
    MockUsageMetadata,
)


# Test environment markers
def pytest_configure(config):
    """Configure custom markers for test organization."""
    markers = [
        "unit: Fast, isolated unit tests",
        "integration: Component integration tests with mocked APIs",
        "api: Real API integration tests (requires API key)",
        "regression: Regression prevention tests",
        "performance: Performance and efficiency tests",
        "e2e: End-to-end user scenario tests",
        "slow: Tests that take >5 seconds",
        "expensive: Tests that use significant API quota",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


# Environment detection
@pytest.fixture(scope="session")
def test_environment():
    """Detect and configure test environment."""
    env_type = "mock"  # Default to mocked tests

    if os.getenv("GEMINI_API_KEY") and os.getenv("ENABLE_API_TESTS"):
        env_type = "api"
    elif os.getenv("CI"):
        env_type = "ci"

    return {
        "type": env_type,
        "has_api_key": bool(os.getenv("GEMINI_API_KEY")),
        "is_ci": bool(os.getenv("CI")),
        "enable_api_tests": bool(os.getenv("ENABLE_API_TESTS")),
    }


# Skip logic for API tests
def pytest_collection_modifyitems(config, items):
    """Automatically skip API tests when API key unavailable."""
    if not (os.getenv("GEMINI_API_KEY") and os.getenv("ENABLE_API_TESTS")):
        skip_api = pytest.mark.skip(
            reason="API tests require GEMINI_API_KEY and ENABLE_API_TESTS=1"
        )
        for item in items:
            if "api" in item.keywords:
                item.add_marker(skip_api)


# Enhanced fixtures
@pytest.fixture
def mock_api_key():
    """Provide test API key"""
    return "test_api_key_12345_67890_abcdef_ghijkl"


@pytest.fixture
def mock_env(mock_api_key, monkeypatch):
    """Mock environment variables"""
    monkeypatch.setenv("GEMINI_API_KEY", mock_api_key)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")


@pytest.fixture
def mock_genai_client():
    """Mock the underlying genai.Client"""
    with patch("google.genai.Client") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        yield client_instance


@pytest.fixture
def client_config(mock_api_key):
    """Provide ClientConfiguration for testing"""
    return ClientConfiguration(
        api_key=mock_api_key, model_name="gemini-2.0-flash", enable_caching=False
    )


@pytest.fixture
def gemini_client(client_config, mock_genai_client):
    """Provide configured GeminiClient for testing"""
    return GeminiClient(client_config)


@pytest.fixture
def batch_processor(gemini_client):
    """Provide BatchProcessor for testing"""
    return BatchProcessor(client=gemini_client)


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


# Conversation testing fixtures
@pytest.fixture
def conversation_session(batch_processor):
    """Provide ConversationSession for testing"""
    from gemini_batch.conversation import create_conversation

    return create_conversation(
        "Test content for conversation", processor=batch_processor
    )


@pytest.fixture
def multi_source_content():
    """Content with multiple sources for testing"""
    return [
        "Primary source: Machine learning fundamentals",
        "Secondary source: Deep learning applications",
        "Tertiary source: AI ethics and considerations",
    ]


@pytest.fixture
def conversation_questions():
    """Questions that build on each other for conversation testing"""
    return [
        "What is the main topic?",
        "How does this relate to what we discussed?",
        "What are the practical implications?",
        "What should I learn next?",
    ]
