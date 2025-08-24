"""Unit tests for planner hint consumption and token estimate adjustments."""

import pytest

from gemini_batch.config import resolve_config
from gemini_batch.core.types import (
    InitialCommand,
    ResolvedCommand,
    Success,
)
from gemini_batch.pipeline.hints import CacheHint, EstimationOverrideHint
from gemini_batch.pipeline.planner import ExecutionPlanner

pytestmark = pytest.mark.unit


class TestPlannerHintConsumption:
    """Test planner's consumption of CacheHint and EstimationOverrideHint."""

    @pytest.fixture
    def basic_resolved_command(self):
        """Create a basic resolved command for testing."""
        config = resolve_config()
        initial = InitialCommand(
            sources=(),
            prompts=("test prompt",),
            config=config,
        )
        return ResolvedCommand(initial=initial, resolved_sources=())

    @pytest.fixture
    def planner(self):
        """Create a planner instance for testing."""
        return ExecutionPlanner()

    @pytest.mark.asyncio
    async def test_no_hints_preserves_existing_behavior(
        self, planner, basic_resolved_command
    ):
        """Without hints, planner should behave exactly as before."""
        result = await planner.handle(basic_resolved_command)

        assert isinstance(result, Success)
        planned = result.value
        assert planned.token_estimate is not None
        assert (
            planned.execution_plan.explicit_cache is None
        )  # No caching for small prompts

    @pytest.mark.asyncio
    async def test_cache_hint_forces_explicit_cache_plan(self, planner):
        """CacheHint should force creation of ExplicitCachePlan regardless of token count."""
        config = resolve_config()
        initial = InitialCommand(
            sources=(),
            prompts=("short",),  # Small prompt that normally wouldn't cache
            config=config,
            hints=(CacheHint("custom-cache-key"),),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())

        result = await planner.handle(resolved)

        assert isinstance(result, Success)
        planned = result.value
        # Should have explicit cache plan even for small prompt
        assert planned.execution_plan.explicit_cache is not None
        assert (
            planned.execution_plan.explicit_cache.deterministic_key
            == "custom-cache-key"
        )
        assert planned.execution_plan.explicit_cache.create is True

    @pytest.mark.asyncio
    async def test_cache_hint_reuse_only_policy(self, planner):
        """CacheHint with reuse_only=True should set create=False."""
        config = resolve_config()
        initial = InitialCommand(
            sources=(),
            prompts=("test prompt",),
            config=config,
            hints=(CacheHint("reuse-key", reuse_only=True),),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())

        result = await planner.handle(resolved)

        assert isinstance(result, Success)
        planned = result.value
        assert planned.execution_plan.explicit_cache is not None
        assert planned.execution_plan.explicit_cache.create is False
        assert planned.execution_plan.explicit_cache.deterministic_key == "reuse-key"

    @pytest.mark.asyncio
    async def test_cache_hint_ttl_override(self, planner):
        """CacheHint should override TTL from config."""
        config = resolve_config(overrides={"ttl_seconds": 1800})
        initial = InitialCommand(
            sources=(),
            prompts=("test prompt",),
            config=config,
            hints=(CacheHint("test-key", ttl_seconds=7200),),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())

        result = await planner.handle(resolved)

        assert isinstance(result, Success)
        planned = result.value
        assert planned.execution_plan.explicit_cache is not None
        assert (
            planned.execution_plan.explicit_cache.ttl_seconds == 7200
        )  # From hint, not config

    @pytest.mark.asyncio
    async def test_estimation_override_widen_factor(
        self, planner, basic_resolved_command
    ):
        """EstimationOverrideHint should multiply max_tokens by widen_max_factor."""
        # First get baseline estimate
        baseline_result = await planner.handle(basic_resolved_command)
        assert isinstance(baseline_result, Success)
        baseline_max = baseline_result.value.token_estimate.max_tokens

        # Now test with widen factor
        config = resolve_config()
        initial = InitialCommand(
            sources=(),
            prompts=("test prompt",),
            config=config,
            hints=(EstimationOverrideHint(widen_max_factor=2.0),),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())

        result = await planner.handle(resolved)

        assert isinstance(result, Success)
        planned = result.value
        # Should be approximately doubled (allowing for slight variations in estimation)
        assert planned.token_estimate.max_tokens >= baseline_max * 1.8
        assert planned.token_estimate.max_tokens <= baseline_max * 2.2

    @pytest.mark.asyncio
    async def test_estimation_override_clamp_max_tokens(self, planner):
        """EstimationOverrideHint should clamp max_tokens when specified."""
        config = resolve_config()
        initial = InitialCommand(
            sources=(),
            prompts=("test prompt",),
            config=config,
            hints=(
                EstimationOverrideHint(widen_max_factor=10.0, clamp_max_tokens=100),
            ),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())

        result = await planner.handle(resolved)

        assert isinstance(result, Success)
        planned = result.value
        # Should be clamped to 100 despite large widen factor
        assert planned.token_estimate.max_tokens == 100

    @pytest.mark.asyncio
    async def test_estimation_override_preserves_other_fields(self, planner):
        """EstimationOverrideHint should only modify max_tokens, preserve others."""
        config = resolve_config()
        initial = InitialCommand(
            sources=(),
            prompts=("test prompt",),
            config=config,
            hints=(EstimationOverrideHint(widen_max_factor=2.0),),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())

        result = await planner.handle(resolved)

        assert isinstance(result, Success)
        planned = result.value
        estimate = planned.token_estimate
        # Other fields should be preserved
        assert estimate.min_tokens >= 0
        assert estimate.expected_tokens >= 0
        assert 0.0 <= estimate.confidence <= 1.0
        # breakdown can be None in some estimation adapters

    @pytest.mark.asyncio
    async def test_multiple_hints_processed_correctly(self, planner):
        """Multiple hints should be processed independently."""
        config = resolve_config()
        initial = InitialCommand(
            sources=(),
            prompts=("test prompt",),
            config=config,
            hints=(
                CacheHint("multi-key", ttl_seconds=1800),
                EstimationOverrideHint(widen_max_factor=1.5, clamp_max_tokens=5000),
            ),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())

        result = await planner.handle(resolved)

        assert isinstance(result, Success)
        planned = result.value

        # Cache hint should be applied
        assert planned.execution_plan.explicit_cache is not None
        assert planned.execution_plan.explicit_cache.deterministic_key == "multi-key"
        assert planned.execution_plan.explicit_cache.ttl_seconds == 1800

        # Estimation override should be applied
        assert planned.token_estimate.max_tokens <= 5000

    @pytest.mark.asyncio
    async def test_unknown_hint_types_ignored_safely(self, planner):
        """Unknown hint types should be silently ignored."""
        config = resolve_config()

        # Create a mock unknown hint type
        class UnknownHint:
            def __init__(self):
                self.unknown_field = "ignored"

        initial = InitialCommand(
            sources=(),
            prompts=("test prompt",),
            config=config,
            hints=(UnknownHint(), CacheHint("known-key")),  # type: ignore
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())

        result = await planner.handle(resolved)

        # Should succeed and process known hints
        assert isinstance(result, Success)
        planned = result.value
        assert planned.execution_plan.explicit_cache is not None
        assert planned.execution_plan.explicit_cache.deterministic_key == "known-key"

    @pytest.mark.asyncio
    async def test_empty_hints_tuple_equivalent_to_none(self, planner):
        """Empty hints tuple should behave identically to hints=None."""
        config = resolve_config()

        # Command with empty hints
        initial_empty = InitialCommand(
            sources=(), prompts=("test prompt",), config=config, hints=()
        )
        resolved_empty = ResolvedCommand(initial=initial_empty, resolved_sources=())

        # Command with no hints
        initial_none = InitialCommand(
            sources=(), prompts=("test prompt",), config=config, hints=None
        )
        resolved_none = ResolvedCommand(initial=initial_none, resolved_sources=())

        result_empty = await planner.handle(resolved_empty)
        result_none = await planner.handle(resolved_none)

        assert isinstance(result_empty, Success)
        assert isinstance(result_none, Success)

        # Token estimates should be identical
        assert (
            result_empty.value.token_estimate.max_tokens
            == result_none.value.token_estimate.max_tokens
        )

        # Cache decisions should be identical
        cache_empty = result_empty.value.execution_plan.explicit_cache
        cache_none = result_none.value.execution_plan.explicit_cache
        assert (cache_empty is None) == (cache_none is None)
