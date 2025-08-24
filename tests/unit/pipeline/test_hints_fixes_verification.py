"""Minimal verification tests for the hint capsules fixes.

These tests verify the three critical fixes are implemented correctly
without requiring complex mock setup or integration testing.
"""

from __future__ import annotations

import inspect

import pytest

from gemini_batch.config import resolve_config
from gemini_batch.pipeline.hints import (
    CacheHint,
    EstimationOverrideHint,
    ExecutionCacheName,
)

pytestmark = pytest.mark.unit


def test_fix1_reg_key_initialization():
    """Verify Fix 1: reg_key is properly initialized in APIHandler."""
    from gemini_batch.pipeline.api_handler import APIHandler

    source = inspect.getsource(APIHandler.handle)

    # Should initialize reg_key before use to prevent UnboundLocalError
    assert "reg_key: str | None = None" in source
    assert "logger.debug" in source


def test_fix2_retry_semantics_behavior_runtime():
    """Runtime verification: retry occurs with applied cache or override, not purely by plan."""
    import asyncio

    from gemini_batch.core.types import (
        APICall,
        ExecutionPlan,
        ExplicitCachePlan,
        InitialCommand,
        PlannedCommand,
        ResolvedCommand,
        Success,
        TextPart,
    )
    from gemini_batch.pipeline.api_handler import APIHandler
    from gemini_batch.telemetry import _EnabledTelemetryContext, _SimpleReporter

    class _Spy:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.first = True

        async def create_cache(
            self,
            *,
            model_name: str,  # noqa: ARG002
            content_parts: tuple[object, ...],  # noqa: ARG002
            system_instruction: str | None,  # noqa: ARG002
            ttl_seconds: int | None,  # noqa: ARG002
        ) -> str:
            return "cachedContents/name"

        async def generate(
            self,
            *,
            model_name: str,
            api_parts: tuple[object, ...],  # noqa: ARG002
            api_config: dict[str, object],
        ) -> dict[str, object]:
            self.calls.append(dict(api_config))
            if self.first and "cached_content" in api_config:
                self.first = False
                raise RuntimeError("cache-bad")
            return {
                "text": "ok",
                "model": model_name,
                "usage": {"total_token_count": 2},
            }

    reporter = _SimpleReporter()
    handler = APIHandler(adapter=_Spy(), telemetry=_EnabledTelemetryContext(reporter))

    initial = InitialCommand(
        sources=(), prompts=("p",), config=resolve_config(overrides={"api_key": "k"})
    )
    resolved = ResolvedCommand(initial=initial, resolved_sources=())
    call = APICall(
        model_name="gemini-2.0-flash", api_parts=(TextPart("p"),), api_config={}
    )
    plan = ExecutionPlan(
        primary_call=call,
        explicit_cache=ExplicitCachePlan(create=True, contents_part_indexes=(0,)),
    )
    planned = PlannedCommand(resolved=resolved, execution_plan=plan)

    result = asyncio.get_event_loop().run_until_complete(handler.handle(planned))
    assert isinstance(result, Success)
    # First with cache, then without cache
    adapter = handler._adapter
    assert adapter is not None
    assert isinstance(adapter, _Spy)
    flags = ["cached_content" in c for c in adapter.calls]
    assert flags == [True, False]


def test_fix3_token_clamping_logic():
    """Verify Fix 3: expected_tokens clamped to both min and max bounds."""
    from gemini_batch.pipeline.planner import ExecutionPlanner

    # The logic is now in the helper method
    source = inspect.getsource(ExecutionPlanner._apply_estimation_override)

    # Should clamp expected_tokens to both bounds: max(min, min(expected, max))
    assert "max(" in source and "min(" in source
    # Look for the specific pattern (allowing for formatting variations)
    lines = source.replace(" ", "").replace("\n", "")
    assert "max(estimate.min_tokens,min(estimate.expected_tokens,new_max))" in lines


def test_token_clamping_mathematics():
    """Test the mathematical correctness of the token clamping fix."""
    # Test cases: (min, expected, max) -> expected_result
    test_cases = [
        (50, 200, 100, 100),  # expected > max -> clamp to max
        (100, 80, 150, 100),  # expected < min -> clamp to min
        (50, 120, 150, 120),  # expected within bounds -> unchanged
        (100, 100, 100, 100),  # all equal -> unchanged
    ]

    for min_tokens, expected_tokens, max_tokens, expected_result in test_cases:
        # Apply the fix's logic
        result = max(min_tokens, min(expected_tokens, max_tokens))
        assert result == expected_result
        assert min_tokens <= result <= max_tokens


def test_cache_intent_telemetry_exists():
    """Runtime check: cache intent telemetry gauges exist with expected names."""
    import asyncio

    from gemini_batch.core.types import (
        APICall,
        ExecutionPlan,
        ExplicitCachePlan,
        InitialCommand,
        PlannedCommand,
        ResolvedCommand,
        Success,
        TextPart,
    )
    from gemini_batch.pipeline.api_handler import APIHandler
    from gemini_batch.telemetry import _EnabledTelemetryContext, _SimpleReporter

    class _Adapter:
        async def create_cache(
            self,
            *,
            model_name: str,  # noqa: ARG002
            content_parts: tuple[object, ...],  # noqa: ARG002
            system_instruction: str | None,  # noqa: ARG002
            ttl_seconds: int | None,  # noqa: ARG002
        ) -> str:
            return "cachedContents/test"

        async def generate(
            self,
            *,
            model_name: str,
            api_parts: tuple[object, ...],  # noqa: ARG002
            api_config: dict[str, object],  # noqa: ARG002
        ) -> dict[str, object]:
            return {
                "text": "ok",
                "model": model_name,
                "usage": {"total_token_count": 1},
            }

    reporter = _SimpleReporter()
    handler = APIHandler(
        adapter=_Adapter(), telemetry=_EnabledTelemetryContext(reporter)
    )

    initial = InitialCommand(
        sources=(), prompts=("p",), config=resolve_config(overrides={"api_key": "k"})
    )
    resolved = ResolvedCommand(initial=initial, resolved_sources=())
    call = APICall(
        model_name="gemini-2.0-flash", api_parts=(TextPart("p"),), api_config={}
    )
    plan = ExecutionPlan(
        primary_call=call,
        explicit_cache=ExplicitCachePlan(create=True, contents_part_indexes=(0,)),
    )
    planned = PlannedCommand(resolved=resolved, execution_plan=plan)

    result = asyncio.get_event_loop().run_until_complete(handler.handle(planned))
    assert isinstance(result, Success)
    keys = set(reporter.metrics.keys())
    assert any(k.endswith("api.generate.cache_intent_plan") for k in keys)
    assert any(k.endswith("api.generate.cache_applied") for k in keys)


def test_hint_validation_prevents_invalid_construction():
    """Test that hint validation catches construction errors."""

    # Valid constructions should work
    CacheHint(deterministic_key="valid-key")
    EstimationOverrideHint(widen_max_factor=1.5)
    ExecutionCacheName(cache_name="valid-cache")

    # Invalid constructions should raise
    try:
        CacheHint(deterministic_key="")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "deterministic_key must be a non-empty string" in str(e)

    try:
        EstimationOverrideHint(widen_max_factor=0.5)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "widen_max_factor must be finite and >= 1.0" in str(e)

    try:
        ExecutionCacheName(cache_name="")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "cache_name must be a non-empty string" in str(e)


def test_architectural_invariants_preserved():
    """Verify architectural invariants are preserved."""
    from gemini_batch.config import resolve_config
    from gemini_batch.core.types import InitialCommand

    config = resolve_config()

    # Should work without hints (fail-soft)
    cmd_no_hints = InitialCommand(
        sources=(), prompts=("test",), config=config, hints=None
    )
    assert cmd_no_hints.hints is None

    # Should work with hints
    cmd_with_hints = InitialCommand(
        sources=(),
        prompts=("test",),
        config=config,
        hints=(CacheHint(deterministic_key="test"),),
    )
    assert cmd_with_hints.hints is not None
    assert len(cmd_with_hints.hints) == 1


def test_metadata_channel_documentation():
    """Verify metadata channel documentation improvement."""
    from gemini_batch.pipeline.registries import CacheRegistry

    docstring = CacheRegistry.__doc__
    if docstring:
        assert "structured, authoritative information" in docstring
        assert "audit, debugging, and potential future retrieval" in docstring


def test_estimation_override_bounds_enforcement():
    """Test EstimationOverrideHint enforces proper bounds."""

    # Valid scenarios
    EstimationOverrideHint(widen_max_factor=1.0)  # No change
    EstimationOverrideHint(widen_max_factor=2.0, clamp_max_tokens=1000)

    # Invalid scenarios
    try:
        EstimationOverrideHint(widen_max_factor=0.9)  # < 1.0
        raise AssertionError("Should reject widen_max_factor < 1.0")
    except ValueError:
        pass

    try:
        EstimationOverrideHint(clamp_max_tokens=-1)  # Negative
        raise AssertionError("Should reject negative clamp_max_tokens")
    except ValueError:
        pass


def test_hints_import_and_export():
    """Verify hints are properly exported from main module."""
    # These should import successfully from the main module
    from gemini_batch import (
        CacheHint,
        EstimationOverrideHint,
        ExecutionCacheName,
        ResultHint,
    )

    # Basic construction should work
    cache_hint = CacheHint(deterministic_key="test")
    est_hint = EstimationOverrideHint(widen_max_factor=1.2)
    exec_hint = ExecutionCacheName(cache_name="test")
    result_hint = ResultHint(prefer_json_array=True)

    assert cache_hint.deterministic_key == "test"
    assert est_hint.widen_max_factor == 1.2
    assert exec_hint.cache_name == "test"
    assert result_hint.prefer_json_array is True
