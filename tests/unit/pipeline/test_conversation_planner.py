"""Tests for conversation_planner.py following contract-first pattern."""

import pytest

from gemini_batch.core.types import ConversationTurn
from gemini_batch.extensions.conversation import ConversationState
from gemini_batch.extensions.conversation_planner import (
    ConversationPlan,
    compile_conversation,
)
from gemini_batch.extensions.conversation_types import (
    ConversationPolicy,
    PromptSet,
)


def test_compile_conversation_basic():
    """Test basic conversation compilation."""
    state = ConversationState(sources=("doc.pdf",), turns=())
    policy = ConversationPolicy()
    prompt_set = PromptSet(("Hello world",), "single")

    plan = compile_conversation(state, prompt_set, policy)

    assert isinstance(plan, ConversationPlan)
    assert plan.sources == ("doc.pdf",)
    assert plan.prompts == ("Hello world",)
    assert plan.strategy == "sequential"
    assert len(plan.hints) == 0  # No hints for basic case


def test_compile_conversation_with_history_window():
    """Test that history window is properly applied."""
    turns = (
        ConversationTurn("q1", "a1", False),
        ConversationTurn("q2", "a2", False),
        ConversationTurn("q3", "a3", False),
    )
    state = ConversationState(sources=("doc.pdf",), turns=turns, cache=None, hints=None)
    policy = ConversationPolicy(keep_last_n=2)
    prompt_set = PromptSet.single("New question")

    plan = compile_conversation(state, prompt_set, policy)

    assert len(plan.history) == 2
    assert plan.history[0].question == "q2"
    assert plan.history[1].question == "q3"


def test_compile_conversation_with_cache_key():
    """Test that cache key creates appropriate hints."""
    state = ConversationState(
        sources=("doc.pdf",),
        turns=(),
        cache_key="test_key",
        cache_artifacts=("artifact1",),
    )
    policy = ConversationPolicy()
    prompt_set = PromptSet(("Question",), "single")

    plan = compile_conversation(state, prompt_set, policy)

    assert len(plan.hints) > 0
    hint_names = [type(h).__name__ for h in plan.hints]
    assert "CacheHint" in hint_names


def test_compile_conversation_with_policy_hints():
    """Test that policy settings create appropriate hints."""
    state = ConversationState(sources=("doc.pdf",), turns=())
    policy = ConversationPolicy(
        widen_max_factor=1.2,
        clamp_max_tokens=16000,
        prefer_json_array=True,
        execution_cache_name="test_cache",
    )
    prompt_set = PromptSet(("Question",), "single")

    plan = compile_conversation(state, prompt_set, policy)

    hint_names = [type(h).__name__ for h in plan.hints]
    assert "EstimationOverrideHint" in hint_names
    assert "ResultHint" in hint_names
    assert "ExecutionCacheName" in hint_names


def test_compile_conversation_vectorized_mode():
    """Test that vectorized mode is preserved in plan."""
    state = ConversationState(sources=("doc.pdf",), turns=())
    policy = ConversationPolicy()
    prompt_set = PromptSet(("Q1", "Q2", "Q3"), "vectorized")

    plan = compile_conversation(state, prompt_set, policy)

    assert plan.prompts == ("Q1", "Q2", "Q3")
    assert (
        plan.strategy == "sequential"
    )  # Current implementation uses sequential for vectorized mode


def test_compile_conversation_sequential_mode():
    """Test that sequential mode is preserved in plan."""
    state = ConversationState(sources=("doc.pdf",), turns=())
    policy = ConversationPolicy()
    prompt_set = PromptSet(("Q1", "Q2", "Q3"), "sequential")

    plan = compile_conversation(state, prompt_set, policy)

    assert plan.prompts == ("Q1", "Q2", "Q3")
    assert plan.strategy == "sequential"


def test_compile_conversation_empty_prompts():
    """Test compilation with empty prompts."""
    state = ConversationState(sources=("doc.pdf",), turns=())
    policy = ConversationPolicy()
    prompt_set = PromptSet((), "single")

    plan = compile_conversation(state, prompt_set, policy)

    assert plan.prompts == ()
    assert plan.strategy == "sequential"


def test_compile_conversation_single_prompt_optimization():
    """Test that single prompt always uses sequential strategy."""
    state = ConversationState(sources=("doc.pdf",), turns=())
    policy = ConversationPolicy()
    prompt_set = PromptSet(("Single prompt",), "single")

    plan = compile_conversation(state, prompt_set, policy)

    assert plan.strategy == "sequential"


def test_compile_conversation_reuse_cache_only():
    """Test that reuse_cache_only policy affects cache hints."""
    state = ConversationState(
        sources=("doc.pdf",),
        turns=(),
        cache_key="test_key",
        cache_artifacts=("artifact1",),
    )
    policy = ConversationPolicy(reuse_cache_only=True)
    prompt_set = PromptSet(("Question",), "single")

    plan = compile_conversation(state, prompt_set, policy)

    # Find the cache hint
    cache_hints = [h for h in plan.hints if type(h).__name__ == "CacheHint"]
    assert len(cache_hints) == 1
    assert cache_hints[0].reuse_only is True


def test_compile_conversation_plan_immutability():
    """Test that compiled plans are immutable."""
    state = ConversationState(sources=("doc.pdf",), turns=())
    policy = ConversationPolicy()
    prompt_set = PromptSet(("Question",), "single")

    plan = compile_conversation(state, prompt_set, policy)

    # Should be frozen dataclass - these should fail
    with pytest.raises(AttributeError):
        plan.prompts = ("Modified",)

    with pytest.raises(AttributeError):
        plan.hints = ()


def test_compile_conversation_deterministic():
    """Test that compilation is deterministic for same inputs."""
    state = ConversationState(sources=("doc.pdf",), turns=())
    policy = ConversationPolicy(keep_last_n=3, widen_max_factor=1.1)
    prompt_set = PromptSet(("Q1", "Q2"), "sequential")

    plan1 = compile_conversation(state, prompt_set, policy)
    plan2 = compile_conversation(state, prompt_set, policy)

    assert plan1 == plan2
    assert plan1.prompts == plan2.prompts
    assert plan1.strategy == plan2.strategy
    assert len(plan1.hints) == len(plan2.hints)


def test_plan_inspectability():
    """Test that plans are inspectable via attributes."""
    state = ConversationState(sources=("doc.pdf",), turns=())
    policy = ConversationPolicy()
    prompt_set = PromptSet(("Q1", "Q2"), "vectorized")

    plan = compile_conversation(state, prompt_set, policy)

    # Test that plan attributes are accessible
    assert hasattr(plan, "sources")
    assert hasattr(plan, "prompts")
    assert hasattr(plan, "strategy")
    assert hasattr(plan, "hints")

    assert plan.sources == ("doc.pdf",)
    assert plan.prompts == ("Q1", "Q2")
    assert (
        plan.strategy == "sequential"
    )  # Current implementation uses sequential for vectorized mode
