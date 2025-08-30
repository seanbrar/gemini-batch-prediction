"""Unit tests for conversation extension cache hint integration."""

import pytest

from gemini_batch.config import resolve_config
from gemini_batch.executor import create_executor
from gemini_batch.extensions.conversation import (
    Conversation,
    ConversationState,
)


class TestConversationCache:
    """Test conversation extension cache functionality."""

    @pytest.fixture
    def executor(self):
        """Create executor for conversation testing."""
        config = resolve_config()
        return create_executor(config)

    @pytest.mark.asyncio
    async def test_conversation_with_cache_key(self, executor):
        """Test that conversation works with cache key in state."""
        # Create conversation state with cache key
        state = ConversationState(
            sources=("doc.pdf",),
            turns=(),
            cache_key="test-cache-key",
            cache_artifacts=("artifact1",),
        )
        conv = Conversation(state, executor)

        # Test that cache key is preserved
        assert conv.state.cache_key == "test-cache-key"
        assert conv.state.cache_artifacts == ("artifact1",)

        # Test that conversation can still function
        conv2 = await conv.ask("test question")
        assert len(conv2.state.turns) == 1

    @pytest.mark.asyncio
    async def test_conversation_without_cache_key(self, executor):
        """Test conversation works without cache key."""
        conv = Conversation.start(executor, sources=("doc.pdf",))

        # Test that cache key is None by default
        assert conv.state.cache_key is None
        assert conv.state.cache_artifacts == ()

        # Test that conversation can still function
        conv2 = await conv.ask("test question")
        assert len(conv2.state.turns) == 1
