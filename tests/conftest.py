"""
Global test configuration with support for different test types.
"""

import os
from unittest.mock import MagicMock, PropertyMock, patch

from pydantic import BaseModel
import pytest

from gemini_batch import BatchProcessor, GeminiClient

# --- Test Environment Markers ---


def pytest_configure(config):  # noqa: ANN001, ANN201
    """Configure custom markers for test organization."""
    markers = [
        "unit: Fast, isolated unit tests",
        "integration: Component integration tests with mocked APIs",
        "api: Real API integration tests (requires API key)",
        "characterization: Golden master tests to detect behavior changes.",
        "slow: Tests that take >1 second",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):  # noqa: ANN001, ANN201, ARG001
    """Automatically skip API tests when API key is unavailable."""
    if not (os.getenv("GEMINI_API_KEY") and os.getenv("ENABLE_API_TESTS")):
        skip_api = pytest.mark.skip(
            reason="API tests require GEMINI_API_KEY and ENABLE_API_TESTS=1"  # noqa: COM812
        )
        for item in items:
            if "api" in item.keywords:
                item.add_marker(skip_api)


# --- Core Fixtures ---


@pytest.fixture
def mock_api_key():  # noqa: ANN201
    """Provide a consistent, fake API key for tests."""
    return "test_api_key_12345_67890_abcdef_ghijkl"


@pytest.fixture
def mock_env(mock_api_key, monkeypatch):  # noqa: ANN001, ANN201
    """
    Mocks essential environment variables to ensure tests run in a
    consistent, isolated environment.
    """
    monkeypatch.setenv("GEMINI_API_KEY", mock_api_key)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("GEMINI_ENABLE_CACHING", "False")


@pytest.fixture
def mock_gemini_client(mock_env):  # noqa: ANN001, ANN201, ARG001
    """
    Provides a MagicMock of the GeminiClient.

    This is a powerful fixture that prevents real API calls. The mock's methods,
    like `generate_content`, can be configured within individual tests to return
    specific, predictable responses, making tests deterministic and fast.
    """
    # We create a mock instance that has the same interface as the real GeminiClient.
    mock_client = MagicMock(spec=GeminiClient)

    # We mock the `generate_content` method, as this is the primary method
    # called by BatchProcessor.
    mock_client.generate_content.return_value = {
        "text": '["Default mock answer"]',
        "usage_metadata": {
            "prompt_tokens": 10,
            "candidates_token_count": 5,
            "total_tokens": 15,
        },
    }
    return mock_client


@pytest.fixture
def batch_processor(mock_gemini_client):  # noqa: ANN001, ANN201
    """
    Provides a BatchProcessor instance that is pre-configured with a
    mocked GeminiClient. This is the standard way to test BatchProcessor
    without hitting the actual API.
    """
    # We inject the mock client directly into the processor.
    # This is a key principle of Dependency Injection for testability.
    return BatchProcessor(_client=mock_gemini_client)


# --- Advanced Fixtures for Client Behavior Testing ---


@pytest.fixture
def mocked_internal_genai_client():  # noqa: ANN201
    """
    Mocks the internal `google.genai.Client` that GeminiClient uses.

    This allows us to test the logic of our GeminiClient by inspecting the
    low-level API calls it attempts to make, without any real network activity.
    """
    # We patch the class that our client will instantiate.
    with patch("gemini_batch.gemini_client.genai.Client") as mock_genai:
        # Create an instance of the mock to be used by our client
        mock_instance = mock_genai.return_value

        # Mock the nested structure for creating and using caches
        mock_instance.caches.create.return_value = MagicMock(
            name="caches.create_return"  # noqa: COM812
        )
        type(mock_instance.caches.create.return_value).name = PropertyMock(
            return_value="cachedContents/mock-cache-123"  # noqa: COM812
        )

        # Mock the token counter to avoid real API calls during planning
        mock_instance.models.count_tokens.return_value = MagicMock(total_tokens=5000)

        yield mock_instance


@pytest.fixture
def caching_gemini_client(mock_env, mocked_internal_genai_client):  # noqa: ANN001, ANN201, ARG001
    """
    Provides a real GeminiClient instance configured for caching, but with
    its internal API calls mocked.
    """
    # We instantiate our actual GeminiClient. Because the `genai.Client` is
    # patched, our client will receive the `mocked_internal_genai_client`
    # when it tries to initialize its internal client.
    # We explicitly enable caching for this test client.
    client = GeminiClient(enable_caching=True)
    return client  # noqa: RET504


@pytest.fixture
def mock_httpx_client():  # noqa: ANN201
    """
    Mocks the httpx.Client to prevent real network requests for URL processing.
    """
    # We patch the client where it's used in the extractors module.
    with patch("gemini_batch.files.extractors.httpx.Client") as mock_client_class:
        mock_instance = mock_client_class.return_value.__enter__.return_value
        yield mock_instance


@pytest.fixture
def mock_get_mime_type():  # noqa: ANN201
    """
    Mocks the get_mime_type utility function to prevent python-magic
    from bypassing pyfakefs, making MIME type detection deterministic in tests.
    """
    # We patch the function in the `utils` module where it is defined.
    with patch("gemini_batch.files.utils.get_mime_type") as mock_func:
        # Define a simple side effect to simulate MIME type detection based on file extension.  # noqa: E501
        def side_effect(file_path, use_magic=True):  # noqa: ANN001, ANN202, ARG001, FBT002
            if str(file_path).endswith(".png"):
                return "image/png"
            if str(file_path).endswith(".txt") or str(file_path).endswith(".md"):
                return "text/plain"
            # Provide a sensible default for any other file types in tests.
            return "application/octet-stream"

        mock_func.side_effect = side_effect
        yield mock_func


# --- Helper Fixtures ---
@pytest.fixture
def fs(fs):  # noqa: ANN001, ANN201
    """
    A fixture for pyfakefs that automatically enables OS-specific path separators.
    This makes filesystem tests more robust across different operating systems.
    """
    fs.os = os
    return fs


# --- Pydantic Schema for Structured Tests ---


class SimpleSummary(BaseModel):
    """A simple Pydantic schema for structured output tests."""

    summary: str
    key_points: list[str]
