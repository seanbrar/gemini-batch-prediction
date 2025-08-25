"""Unit tests for advanced conversation extension features."""

import pytest

from gemini_batch.config import resolve_config
from gemini_batch.executor import create_executor
from gemini_batch.extensions.conversation import (
    Conversation,
    ConversationAnalytics,
    ConversationHints,
    ConversationState,
    Exchange,
)


class TestAdvancedHints:
    """Test advanced hint system integration."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        config = resolve_config()
        return create_executor(config)

    @pytest.mark.asyncio
    async def test_with_hints_creates_proper_hints_object(self, executor):
        """Test that with_hints creates proper ConversationHints."""
        conv = Conversation.start(executor, sources=())

        conv_with_hints = conv.with_hints(
            widen_max_factor=1.5,
            clamp_max_tokens=16000,
            prefer_json_array=True,
            execution_cache_name="custom-cache",
        )

        assert conv_with_hints.state.hints is not None
        assert conv_with_hints.state.hints.widen_max_factor == 1.5
        assert conv_with_hints.state.hints.clamp_max_tokens == 16000
        assert conv_with_hints.state.hints.prefer_json_array is True
        assert conv_with_hints.state.hints.execution_cache_name == "custom-cache"

        # Original conversation unchanged
        assert conv.state.hints is None

    @pytest.mark.asyncio
    async def test_without_hints_removes_hints(self, executor):
        """Test that without_hints removes hints."""
        conv = Conversation.start(executor, sources=())
        conv_with_hints = conv.with_hints(prefer_json_array=True)
        conv_without_hints = conv_with_hints.without_hints()

        assert conv_with_hints.state.hints is not None
        assert conv_without_hints.state.hints is None

    @pytest.mark.asyncio
    async def test_hints_preserved_through_conversation(self, executor):
        """Test that hints are preserved through conversation turns."""
        conv = Conversation.start(executor, sources=())
        conv_with_hints = conv.with_hints(prefer_json_array=True)

        conv_after_ask = await conv_with_hints.ask("test question")

        assert conv_after_ask.state.hints is not None
        assert conv_after_ask.state.hints.prefer_json_array is True


class TestConversationAnalytics:
    """Test conversation analytics and observability."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        config = resolve_config()
        return create_executor(config)

    def test_analytics_empty_conversation(self, executor):
        """Test analytics for empty conversation."""
        conv = Conversation.start(executor, sources=())
        analytics = conv.analytics()

        assert isinstance(analytics, ConversationAnalytics)
        assert analytics.total_turns == 0
        assert analytics.error_turns == 0
        assert analytics.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_analytics_with_conversation(self, executor):
        """Test analytics with actual conversation turns."""
        conv = Conversation.start(executor, sources=())
        conv = await conv.ask("Question 1")
        conv = await conv.ask("Question 2")

        analytics = conv.analytics()

        assert analytics.total_turns == 2
        assert analytics.error_turns == 0
        assert analytics.success_rate == 1.0
        assert analytics.avg_response_length > 0
        assert analytics.total_user_chars > 0
        assert analytics.total_assistant_chars > 0

    def test_health_score_calculation(self, executor):
        """Test health score calculation."""
        conv = Conversation.start(executor, sources=())

        # Empty conversation should have good health
        health = conv.health_score()
        assert 0.0 <= health <= 1.0

    def test_summary_stats_format(self, executor):
        """Test summary stats format."""
        conv = Conversation.start(executor, sources=())
        stats = conv.summary_stats()

        assert isinstance(stats, dict)
        assert "turns" in stats
        assert "success_rate" in stats
        assert "health_score" in stats
        assert "cache_enabled" in stats
        assert "hints_enabled" in stats


class TestConversationBranching:
    """Test conversation branching and forking."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        config = resolve_config()
        return create_executor(config)

    @pytest.mark.asyncio
    async def test_fork_creates_independent_copy(self, executor):
        """Test that fork creates independent conversation copy."""
        conv = Conversation.start(executor, sources=())
        conv = await conv.ask("Initial question")

        fork = conv.fork()
        fork = await fork.ask("Fork question")

        # Original unchanged
        assert len(conv.state.turns) == 1
        assert conv.state.turns[0].user == "Initial question"

        # Fork has additional turn
        assert len(fork.state.turns) == 2
        assert fork.state.turns[1].user == "Fork question"

    def test_rollback_negative_indexing(self, executor):
        """Test rollback with negative indexing."""
        # Create conversation with mock exchanges
        exchanges = (
            Exchange("Q1", "A1", error=False),
            Exchange("Q2", "A2", error=False),
            Exchange("Q3", "A3", error=False),
        )
        state = ConversationState(sources=(), turns=exchanges)
        conv = Conversation(state, executor)

        # Rollback last turn
        rolled_back = conv.rollback(-1)
        assert len(rolled_back.state.turns) == 2

        # Rollback last 2 turns
        rolled_back_2 = conv.rollback(-2)
        assert len(rolled_back_2.state.turns) == 1

    def test_rollback_positive_indexing(self, executor):
        """Test rollback with positive indexing."""
        exchanges = (
            Exchange("Q1", "A1", error=False),
            Exchange("Q2", "A2", error=False),
            Exchange("Q3", "A3", error=False),
        )
        state = ConversationState(sources=(), turns=exchanges)
        conv = Conversation(state, executor)

        # Keep first 2 turns
        rolled_back = conv.rollback(1)
        assert len(rolled_back.state.turns) == 2

    @pytest.mark.asyncio
    async def test_explore_alternatives(self, executor):
        """Test exploring alternative conversation paths."""
        conv = Conversation.start(executor, sources=())
        conv = await conv.ask("Initial question")

        original, alternatives = await conv.explore_alternatives(
            ["Alternative 1", "Alternative 2"]
        )

        # Original unchanged
        assert len(original.state.turns) == 1

        # Alternatives created
        assert len(alternatives) == 2
        assert "Alternative 1" in alternatives
        assert "Alternative 2" in alternatives

        for prompt, (branch, response) in alternatives.items():
            assert len(branch.state.turns) == 2
            assert branch.state.turns[1].user == prompt
            assert isinstance(response, str)

    def test_merge_turns_strategies(self, executor):
        """Test different merge strategies."""
        # Create two conversations with different turns
        exchanges1 = (Exchange("Q1", "A1", error=False),)
        exchanges2 = (Exchange("Q2", "A2", error=False),)

        state1 = ConversationState(sources=(), turns=exchanges1)
        state2 = ConversationState(sources=(), turns=exchanges2)

        conv1 = Conversation(state1, executor)
        conv2 = Conversation(state2, executor)

        # Test append strategy
        merged_append = conv1.merge_turns(conv2, strategy="append")
        assert len(merged_append.state.turns) == 2
        assert merged_append.state.turns[0].user == "Q1"
        assert merged_append.state.turns[1].user == "Q2"


class TestContextManagement:
    """Test advanced context management strategies."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        config = resolve_config()
        return create_executor(config)

    def test_sliding_window(self, executor):
        """Test sliding window context management."""
        # Create conversation with 5 turns
        exchanges = tuple(Exchange(f"Q{i}", f"A{i}", error=False) for i in range(5))
        state = ConversationState(sources=(), turns=exchanges)
        conv = Conversation(state, executor)

        # Apply sliding window of size 3
        windowed = conv.with_sliding_window(3)

        assert len(windowed.state.turns) == 3
        # Should keep last 3 turns
        assert windowed.state.turns[0].user == "Q2"
        assert windowed.state.turns[2].user == "Q4"

    def test_sliding_window_smaller_than_window(self, executor):
        """Test sliding window when conversation is smaller than window."""
        exchanges = (Exchange("Q1", "A1", error=False),)
        state = ConversationState(sources=(), turns=exchanges)
        conv = Conversation(state, executor)

        windowed = conv.with_sliding_window(5)

        # Should return unchanged conversation
        assert len(windowed.state.turns) == 1
        assert windowed.state.turns[0].user == "Q1"

    def test_prune_context_keep_recent(self, executor):
        """Test context pruning with keep_recent strategy."""
        exchanges = tuple(Exchange(f"Q{i}", f"A{i}", error=False) for i in range(5))
        state = ConversationState(sources=(), turns=exchanges)
        conv = Conversation(state, executor)

        pruned = conv.prune_context(strategy="keep_recent", target_turns=3)

        assert len(pruned.state.turns) == 3
        assert pruned.state.turns[0].user == "Q2"

    def test_prune_context_remove_errors(self, executor):
        """Test context pruning with remove_errors strategy."""
        exchanges = (
            Exchange("Q1", "A1", error=False),
            Exchange("Q2", "Error", error=True),  # Error turn
            Exchange("Q3", "A3", error=False),
            Exchange("Q4", "Error", error=True),  # Error turn
            Exchange("Q5", "A5", error=False),
        )
        state = ConversationState(sources=(), turns=exchanges)
        conv = Conversation(state, executor)

        pruned = conv.prune_context(strategy="remove_errors", target_turns=3)

        # Should remove error turns
        assert len(pruned.state.turns) == 3
        for turn in pruned.state.turns:
            assert not turn.error

    def test_context_size_estimate(self, executor):
        """Test context size estimation."""
        exchanges = (
            Exchange("Short question", "Short answer", error=False),
            Exchange("Much longer question here", "Much longer answer here", error=False),
        )
        state = ConversationState(sources=(), turns=exchanges)
        conv = Conversation(state, executor)

        size = conv.context_size_estimate()

        assert isinstance(size, dict)
        assert "turns" in size
        assert "total_characters" in size
        assert "estimated_tokens" in size
        assert size["turns"] == 2
        assert size["total_characters"] > 0


class TestErrorHandling:
    """Test error handling and recovery mechanisms."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        config = resolve_config()
        return create_executor(config)

    def test_recover_from_errors_remove_strategy(self, executor):
        """Test error recovery with remove strategy."""
        exchanges = (
            Exchange("Q1", "A1", error=False),
            Exchange("Q2", "Error", error=True),  # Error turn
            Exchange("Q3", "A3", error=False),
        )
        state = ConversationState(sources=(), turns=exchanges)
        conv = Conversation(state, executor)

        recovered = conv.recover_from_errors(strategy="remove")

        assert len(recovered.state.turns) == 2
        for turn in recovered.state.turns:
            assert not turn.error

    def test_recover_from_errors_mark_strategy(self, executor):
        """Test error recovery with mark strategy."""
        exchanges = (
            Exchange("Q1", "A1", error=False),
            Exchange("Q2", "Error message", error=True),
        )
        state = ConversationState(sources=(), turns=exchanges)
        conv = Conversation(state, executor)

        recovered = conv.recover_from_errors(strategy="mark")

        assert len(recovered.state.turns) == 2
        assert not recovered.state.turns[1].error  # Marked as recovered
        assert "[RECOVERED ERROR:" in recovered.state.turns[1].assistant

    def test_validate_state_healthy(self, executor):
        """Test state validation for healthy conversation."""
        exchanges = (
            Exchange("Q1", "Good answer", error=False),
            Exchange("Q2", "Another good answer", error=False),
        )
        state = ConversationState(sources=(), turns=exchanges)
        conv = Conversation(state, executor)

        validation = conv.validate_state()

        assert isinstance(validation, dict)
        assert validation["is_healthy"] is True
        assert len(validation["issues"]) == 0
        assert validation["error_rate"] == 0.0

    def test_validate_state_with_errors(self, executor):
        """Test state validation with error turns."""
        exchanges = (
            Exchange("Q1", "Good answer", error=False),
            Exchange("Q2", "Error", error=True),
        )
        state = ConversationState(sources=(), turns=exchanges)
        conv = Conversation(state, executor)

        validation = conv.validate_state()

        assert validation["is_healthy"] is False
        assert len(validation["issues"]) > 0
        assert validation["error_rate"] == 0.5


class TestVersioning:
    """Test conversation state versioning."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        config = resolve_config()
        return create_executor(config)

    def test_version_increments_with_changes(self, executor):
        """Test that version increments with state changes."""
        conv = Conversation.start(executor, sources=())
        initial_version = conv.state.version

        # Add hints - should increment version
        conv_with_hints = conv.with_hints(prefer_json_array=True)
        assert conv_with_hints.state.version == initial_version + 1

        # Add cache - should increment version
        conv_with_cache = conv_with_hints.with_cache(key="test")
        assert conv_with_cache.state.version == initial_version + 2

        # Fork - should preserve version
        fork = conv_with_cache.fork()
        assert fork.state.version == conv_with_cache.state.version

    @pytest.mark.asyncio
    async def test_version_increments_with_conversation(self, executor):
        """Test that version increments with conversation turns."""
        conv = Conversation.start(executor, sources=())
        initial_version = conv.state.version

        conv_after_ask = await conv.ask("test question")
        assert conv_after_ask.state.version == initial_version + 1
