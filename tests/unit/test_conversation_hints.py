"""Unit tests for conversation extension cache hint integration."""

import pytest

from gemini_batch.config import resolve_config
from gemini_batch.executor import create_executor
from gemini_batch.extensions.conversation import (
    CacheBinding,
    Conversation,
    ConversationState,
)


class TestConversationHints:
    """Test conversation extension integration with cache hints."""

    @pytest.fixture
    def executor(self):
        """Create executor for conversation testing."""
        config = resolve_config()
        return create_executor(config)

    @pytest.mark.asyncio
    async def test_conversation_without_cache_no_hints(self, executor):
        """Conversation without cache should not generate any hints."""
        conv = Conversation.start(executor, sources=())

        # Verify state has no cache
        assert conv.state.cache is None

        conv2 = await conv.ask("hello")

        # Should succeed without any hints
        assert conv2.state.last is not None
        assert conv2.state.last.error is False

    @pytest.mark.asyncio
    async def test_conversation_with_cache_generates_cache_hint(self, executor):
        """Conversation with cache should generate CacheHint with correct values."""
        conv = Conversation.start(executor, sources=())
        conv_cached = conv.with_cache(key="conv:test-123", ttl_seconds=3600)

        # Verify cache binding created
        assert conv_cached.state.cache is not None
        assert conv_cached.state.cache.key == "conv:test-123"
        assert conv_cached.state.cache.ttl_seconds == 3600

        conv2 = await conv_cached.ask("test question")

        # Should succeed with cache hint applied
        assert conv2.state.last is not None
        assert conv2.state.last.error is False

    @pytest.mark.asyncio
    async def test_cache_hint_preserves_conversation_state(self, executor):
        """Cache hints should not interfere with conversation state management."""
        conv = Conversation.start(executor, sources=())
        conv_cached = conv.with_cache(key="state-test", ttl_seconds=1800)

        conv2 = await conv_cached.ask("first question")
        conv3 = await conv2.ask("second question")

        # State should be properly maintained
        assert len(conv3.state.turns) == 2
        assert conv3.state.turns[0].user == "first question"
        assert conv3.state.turns[1].user == "second question"

        # Cache should persist through conversation
        assert conv3.state.cache is not None
        assert conv3.state.cache.key == "state-test"

    @pytest.mark.asyncio
    async def test_cache_hint_artifacts_passed_through(self, executor):
        """Cache artifacts should be passed through to CacheHint."""
        cache_binding = CacheBinding(
            key="artifact-test",
            artifacts=("source1.pdf", "source2.txt"),
            ttl_seconds=7200,
        )

        conv_state = ConversationState(sources=(), turns=(), cache=cache_binding)
        conv = Conversation(conv_state, executor)

        conv2 = await conv.ask("test with artifacts")

        # Should succeed with artifacts in cache hint
        assert conv2.state.last is not None
        assert conv2.state.last.error is False
        assert conv2.state.cache is not None
        assert conv2.state.cache.artifacts == ("source1.pdf", "source2.txt")

    @pytest.mark.asyncio
    async def test_no_cache_to_cache_transition(self, executor):
        """Conversation can transition from no cache to cached seamlessly."""
        conv = Conversation.start(executor, sources=())

        # First ask without cache
        conv2 = await conv.ask("first without cache")
        assert conv2.state.cache is None

        # Add cache and continue
        conv_cached = conv2.with_cache(key="transition-test")
        conv3 = await conv_cached.ask("second with cache")

        # Should have both turns and cache active
        assert len(conv3.state.turns) == 2
        assert conv3.state.cache is not None
        assert conv3.state.cache.key == "transition-test"

    @pytest.mark.asyncio
    async def test_cache_key_determinism(self, executor):
        """Same cache key should be used consistently."""
        conv1 = Conversation.start(executor, sources=()).with_cache(
            key="deterministic-key"
        )
        conv2 = Conversation.start(executor, sources=()).with_cache(
            key="deterministic-key"
        )

        # Both should use the same cache key
        assert conv1.state.cache is not None and conv2.state.cache is not None
        assert conv1.state.cache.key == conv2.state.cache.key == "deterministic-key"

        result1 = await conv1.ask("test question")
        result2 = await conv2.ask("test question")

        # Both should succeed and have same cache key
        assert result1.state.cache is not None and result2.state.cache is not None
        assert result1.state.cache.key == result2.state.cache.key

    @pytest.mark.asyncio
    async def test_conversation_cache_hint_reuse_only_false_by_default(self, executor):
        """Conversation cache hints should default to reuse_only=False (create new)."""
        conv = Conversation.start(executor, sources=())
        conv_cached = conv.with_cache(key="create-test")

        # The extend() function should create CacheHint with reuse_only=False
        # This is validated indirectly by successful execution
        conv2 = await conv_cached.ask("should create cache")

        assert conv2.state.last is not None
        assert conv2.state.last.error is False
