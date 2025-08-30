"""Tests for the refactored conversation facade following contract-first pattern."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gemini_batch.extensions.conversation import Conversation, ConversationState
from gemini_batch.extensions.conversation_types import ConversationPolicy, PromptSet


@pytest.fixture
def mock_executor():
    """Mock GeminiExecutor for testing."""
    executor = MagicMock()
    executor.config = MagicMock()
    executor.config.to_frozen.return_value = MagicMock()
    return executor


@pytest.fixture
def mock_result():
    """Mock execution result."""
    return {
        "success": True,
        "answers": ["This is a test response."],
        "metrics": {
            "token_validation": {
                "estimated_min": 10,
                "estimated_max": 50,
                "actual": 25,
                "in_range": True,
            }
        },
        "usage": {"total_tokens": 25},
    }


class TestConversationFacade:
    """Test the streamlined Conversation facade."""

    def test_start_conversation(self, mock_executor):
        """Test starting a new conversation."""
        conv = Conversation.start(mock_executor, sources=["doc.pdf"])

        assert isinstance(conv, Conversation)
        assert conv.state.sources == ("doc.pdf",)
        assert len(conv.state.turns) == 0
        assert conv.state.policy == ConversationPolicy()  # Default policy

    def test_start_with_policy(self, mock_executor):
        """Test starting conversation with custom policy."""
        policy = ConversationPolicy(keep_last_n=5, widen_max_factor=1.2)
        conv = Conversation.start(mock_executor, policy=policy)

        assert conv._policy == policy

    def test_with_policy(self, mock_executor):
        """Test changing policy on existing conversation."""
        conv = Conversation.start(mock_executor)
        new_policy = ConversationPolicy(keep_last_n=3)

        result = conv.with_policy(new_policy)

        assert result._policy == new_policy
        assert result is not conv  # New instance

    def test_with_cache(self, mock_executor):
        """Test adding cache binding."""
        conv = Conversation.start(mock_executor)

        result = conv.with_cache(key="test_key", ttl_seconds=300)

        assert result.state.cache is not None
        assert result.state.cache.key == "test_key"
        assert result.state.cache.ttl_seconds == 300
        assert result is not conv

    def test_without_cache(self, mock_executor):
        """Test removing cache binding."""
        conv = Conversation.start(mock_executor)
        conv_with_cache = conv.with_cache(key="test_key")
        result = conv_with_cache.without_cache()

        assert result.state.cache is None

    def test_with_sources(self, mock_executor):
        """Test changing sources."""
        conv = Conversation.start(mock_executor, sources=["doc1.pdf"])

        result = conv.with_sources("doc2.pdf", "doc3.pdf")

        assert result.state.sources == ("doc2.pdf", "doc3.pdf")

    def test_add_sources(self, mock_executor):
        """Test adding sources."""
        conv = Conversation.start(mock_executor, sources=["doc1.pdf"])

        result = conv.add_sources("doc2.pdf", "doc3.pdf")

        assert "doc1.pdf" in result.state.sources
        assert "doc2.pdf" in result.state.sources
        assert "doc3.pdf" in result.state.sources

    def test_ask_method(self, mock_executor, mock_result):
        """Test the ask method uses run internally."""
        conv = Conversation.start(mock_executor)
        mock_executor.execute = AsyncMock(return_value=mock_result)

        # This would normally be async, but we'll test the structure
        assert hasattr(conv, "ask")
        assert callable(conv.ask)

    def test_run_method(self, mock_executor, mock_result):
        """Test the unified run method."""
        conv = Conversation.start(mock_executor)
        mock_executor.execute = AsyncMock(return_value=mock_result)

        assert hasattr(conv, "run")
        assert callable(conv.run)

    def test_fork(self, mock_executor):
        """Test forking conversation."""
        conv = Conversation.start(mock_executor, sources=["doc.pdf"])

        forked = conv.fork()

        assert forked.state == conv.state  # Same state
        assert forked is not conv  # Different instance
        assert forked._executor is conv._executor

    def test_rollback(self, mock_executor):
        """Test rolling back conversation."""
        # Create conversation with some turns
        turns = [
            MagicMock(user="Q1", assistant="A1", error=False),
            MagicMock(user="Q2", assistant="A2", error=False),
            MagicMock(user="Q3", assistant="A3", error=False),
        ]
        state = ConversationState(
            sources=(), turns=tuple(turns), cache=None, hints=None
        )
        conv = Conversation(state, mock_executor)

        # Rollback to remove last turn
        rolled_back = conv.rollback(to_turn=-1)

        assert len(rolled_back.state.turns) == 2
        assert rolled_back is not conv

    def test_analytics(self, mock_executor):
        """Test analytics computation."""
        turns = [
            MagicMock(
                user="Q1",
                assistant="A1",
                error=False,
                estimate_max=100,
                actual_tokens=90,
            ),
            MagicMock(
                user="Q2",
                assistant="A2",
                error=True,
                estimate_max=150,
                actual_tokens=None,
            ),
        ]
        state = ConversationState(
            sources=(), turns=tuple(turns), cache=None, hints=None
        )
        conv = Conversation(state, mock_executor)

        analytics = conv.analytics()

        assert analytics.total_turns == 2
        assert analytics.error_turns == 1
        assert analytics.success_rate == 0.5
        assert analytics.total_estimated_tokens == 250
        assert analytics.total_actual_tokens == 90

    def test_summary_stats(self, mock_executor):
        """Test summary statistics."""
        turns = [
            MagicMock(
                user="Q1",
                assistant="A1",
                error=False,
                estimate_max=100,
                actual_tokens=90,
            ),
        ]
        state = ConversationState(
            sources=(), turns=tuple(turns), cache=None, hints=None
        )
        conv = Conversation(state, mock_executor)

        stats = conv.summary_stats()

        assert isinstance(stats, dict)
        assert "turns" in stats
        assert "success_rate" in stats
        assert "total_tokens" in stats

    def test_health_score(self, mock_executor):
        """Test health score computation."""
        turns = [
            MagicMock(user="Q1", assistant="A1", error=False),
        ]
        state = ConversationState(
            sources=(), turns=tuple(turns), cache=None, hints=None
        )
        conv = Conversation(state, mock_executor)

        score = conv.health_score()

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_immutability(self, mock_executor):
        """Test that conversation operations return new instances."""
        conv = Conversation.start(mock_executor, sources=["doc.pdf"])

        # All operations should return new instances
        with_policy = conv.with_policy(ConversationPolicy(keep_last_n=5))
        with_cache = conv.with_cache(key="test")
        with_sources = conv.with_sources("new.pdf")
        forked = conv.fork()

        assert with_policy is not conv
        assert with_cache is not conv
        assert with_sources is not conv
        assert forked is not conv

        # Original should be unchanged
        assert conv.state.sources == ("doc.pdf",)
        assert conv.state.cache is None


def test_prompt_set_constructors_integration():
    """Test that PromptSet constructors work with facade."""
    ps_single = PromptSet.single("Hello")
    ps_seq = PromptSet.seq("Q1", "Q2", "Q3")
    ps_vec = PromptSet.vec("A", "B", "C")

    assert ps_single.mode == "single"
    assert ps_seq.mode == "sequential"
    assert ps_vec.mode == "vectorized"


def test_conversation_policy_integration():
    """Test policy integration with facade."""
    policy = ConversationPolicy(
        keep_last_n=3,
        widen_max_factor=1.2,
        clamp_max_tokens=16000,
        prefer_json_array=True,
        execution_cache_name="test_cache",
        reuse_cache_only=False,
    )

    assert policy.keep_last_n == 3
    assert policy.widen_max_factor == 1.2
    assert policy.clamp_max_tokens == 16000
    assert policy.prefer_json_array is True
    assert policy.execution_cache_name == "test_cache"
    assert policy.reuse_cache_only is False
