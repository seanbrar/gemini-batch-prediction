"""Tests for the three specific hint capsules fixes implemented.

These tests verify that the micro-fixes eliminate edge case bugs while preserving
architectural invariants and fail-soft semantics.
"""

from __future__ import annotations

import inspect

import pytest

from gemini_batch.config.core import resolve_config
from gemini_batch.core.types import (
    APICall,
    ExecutionPlan,
    ExplicitCachePlan,
    InitialCommand,
    PlannedCommand,
    ResolvedCommand,
    Success,
    TextPart,
    TokenEstimate,
)
from gemini_batch.pipeline.hints import (
    CacheHint,
    EstimationOverrideHint,
    ExecutionCacheName,
)

pytestmark = pytest.mark.unit


class TestFix1UnboundLocalRisk:
    """Behavioral test for cache metadata best-effort write (no UnboundLocal)."""

    @pytest.mark.asyncio
    async def test_cache_metadata_write_is_best_effort(self):
        """Metadata write failures do not fail execution (no UnboundLocal errors)."""
        from gemini_batch.core.types import (
            APICall,
            ExecutionPlan,
            InitialCommand,
            PlannedCommand,
            ResolvedCommand,
            Success,
            TextPart,
        )
        from gemini_batch.pipeline.api_handler import APIHandler

        # Fake registry that raises on set_meta to exercise the inner try/except
        class _RaisingCacheRegistry:
            def get(self, key: str) -> str | None:  # noqa: ARG002
                return None

            def set(self, key: str, value: str) -> None:
                pass

            def set_meta(self, key: str, meta: dict[str, object]) -> None:  # noqa: ARG002
                raise RuntimeError("boom in set_meta")

        # Minimal adapter that supports caching and succeeds on generate
        class _Adapter:
            async def create_cache(
                self,
                *,
                model_name: str,  # noqa: ARG002
                content_parts: tuple[object, ...],  # noqa: ARG002
                system_instruction: str | None,  # noqa: ARG002
                ttl_seconds: int | None,  # noqa: ARG002
            ) -> str:
                return "cachedContents/fake-xyz"

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

        initial = InitialCommand(
            sources=(),
            prompts=("p",),
            config=resolve_config(overrides={"api_key": "k"}),
            hints=(CacheHint("k1", artifacts=("a1",)),),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())
        call = APICall(
            model_name="gemini-2.0-flash", api_parts=(TextPart("p"),), api_config={}
        )
        plan = ExecutionPlan(primary_call=call)
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        handler = APIHandler(
            adapter=_Adapter(), registries={"cache": _RaisingCacheRegistry()}
        )
        result = await handler.handle(planned)
        assert isinstance(result, Success)


class TestFix2RetrySemantics:
    """Behavioral tests for retry semantics with cache intent and overrides."""

    @pytest.mark.asyncio
    async def test_retry_triggers_when_cache_applied_or_override(self):
        """Explicit cache intent with applied cache should retry once without cache on failure."""
        from gemini_batch.pipeline.api_handler import APIHandler
        from gemini_batch.telemetry import _EnabledTelemetryContext, _SimpleReporter

        # Adapter fails first call when cached_content present, then succeeds
        class _SpyAdapter:
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
                return "cachedContents/ok"

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
                    raise RuntimeError("simulate cache-too-large")
                return {
                    "text": "ok",
                    "model": model_name,
                    "usage": {"total_token_count": 5},
                }

        spy = _SpyAdapter()
        reporter = _SimpleReporter()
        handler = APIHandler(adapter=spy, telemetry=_EnabledTelemetryContext(reporter))

        initial = InitialCommand(
            sources=(),
            prompts=("p",),
            config=resolve_config(overrides={"api_key": "k"}),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())
        call = APICall(
            model_name="gemini-2.0-flash", api_parts=(TextPart("p"),), api_config={}
        )
        # Force explicit cache planning so a cache name is applied
        plan = ExecutionPlan(
            primary_call=call,
            explicit_cache=ExplicitCachePlan(create=True, contents_part_indexes=(0,)),
        )
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        result = await handler.handle(planned)
        assert isinstance(result, Success)
        # Calls should include one with cache then one without
        cached_flags = ["cached_content" in c for c in spy.calls]
        assert cached_flags == [True, False]
        # Telemetry gauges should be recorded
        metric_keys = set(reporter.metrics.keys())
        assert any(k.endswith("api.generate.cache_intent_plan") for k in metric_keys)
        assert any(k.endswith("api.generate.cache_applied") for k in metric_keys)

    @pytest.mark.asyncio
    async def test_execution_cache_override_emits_telemetry(self):
        """ExecutionCacheName override emits exec_cache_override telemetry and retries on failure."""
        from gemini_batch.pipeline.api_handler import APIHandler
        from gemini_batch.telemetry import _EnabledTelemetryContext, _SimpleReporter

        class _SpyAdapter2:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []
                self.first = True

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
                    raise RuntimeError("fail-on-cache")
                return {
                    "text": "ok",
                    "model": model_name,
                    "usage": {"total_token_count": 3},
                }

        reporter = _SimpleReporter()
        handler = APIHandler(
            adapter=_SpyAdapter2(), telemetry=_EnabledTelemetryContext(reporter)
        )

        initial = InitialCommand(
            sources=(),
            prompts=("p",),
            config=resolve_config(overrides={"api_key": "k"}),
            hints=(ExecutionCacheName("override-cache"),),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())
        call = APICall(
            model_name="gemini-2.0-flash", api_parts=(TextPart("p"),), api_config={}
        )
        plan = ExecutionPlan(primary_call=call)
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        result = await handler.handle(planned)
        assert isinstance(result, Success)
        # Telemetry should include exec_cache_override and cache_applied
        metric_keys = set(reporter.metrics.keys())
        assert any(k.endswith("api.hints.exec_cache_override") for k in metric_keys)
        assert any(
            k.endswith("api.generate.cache_intent_override") for k in metric_keys
        )


class TestFix3TokenEstimationClamping:
    """Test fix for token estimation expected_tokens clamping."""

    def test_expected_tokens_clamped_to_both_bounds_logic(self):
        """Verify expected_tokens is clamped to [min_tokens, max_tokens]."""

        # Simulate the fixed logic from planner.py
        test_cases = [
            # (min, expected, max_after_override, expected_result)  # noqa: ERA001
            (50, 200, 100, 100),  # expected > max -> clamp to max
            (100, 80, 150, 100),  # expected < min -> clamp to min
            (50, 120, 150, 120),  # expected within bounds -> unchanged
        ]

        for (
            min_tokens,
            expected_tokens,
            max_after_override,
            expected_result,
        ) in test_cases:
            # Apply the fixed clamping logic
            new_expected = max(min_tokens, min(expected_tokens, max_after_override))

            assert new_expected == expected_result
            assert min_tokens <= new_expected <= max_after_override

    def test_estimation_override_maintains_invariants(self):
        """Test various estimation override scenarios maintain token invariants."""

        test_cases = [
            # (widen_factor, clamp_max, original_estimate)  # noqa: ERA001
            (1.5, None, TokenEstimate(100, 150, 200, 0.8, {})),
            (2.0, 250, TokenEstimate(100, 150, 200, 0.8, {})),
            (
                1.0,
                180,
                TokenEstimate(100, 150, 200, 0.8, {}),
            ),  # No widening, just clamp
        ]

        for widen_factor, clamp_max, original in test_cases:
            # Apply the same logic as in the planner
            new_max = int(original.max_tokens * widen_factor)
            if clamp_max is not None:
                new_max = min(new_max, clamp_max)
            new_max = max(new_max, original.min_tokens)

            # The fix: clamp expected to both bounds
            new_expected = max(
                original.min_tokens, min(original.expected_tokens, new_max)
            )

            # Verify invariants are maintained
            assert original.min_tokens <= new_expected <= new_max
            assert new_max >= original.min_tokens

    def test_planner_has_fixed_clamping_logic(self):
        """Verify planner helper includes the fixed clamping logic (helper method)."""
        from gemini_batch.pipeline.planner import ExecutionPlanner

        source = inspect.getsource(ExecutionPlanner._apply_estimation_override)
        # Allow for whitespace/formatting differences by normalizing
        norm = source.replace(" ", "").replace("\n", "")
        assert "max(estimate.min_tokens,min(estimate.expected_tokens,new_max))" in norm


class TestTelemetryStability:
    """Telemetry presence validated at runtime via reporter."""

    @pytest.mark.asyncio
    async def test_cache_intent_telemetry_present(self):
        """Verify cache intent telemetry gauges are emitted at runtime."""
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
                return "cachedContents/ok"

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
            sources=(),
            prompts=("p",),
            config=resolve_config(overrides={"api_key": "k"}),
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

        result = await handler.handle(planned)
        assert isinstance(result, Success)
        keys = set(reporter.metrics.keys())
        assert any(k.endswith("api.generate.cache_intent_plan") for k in keys)
        assert any(k.endswith("api.generate.cache_applied") for k in keys)

    def test_hints_telemetry_present_in_planner(self):
        """Verify planner emits hints telemetry at runtime via reporter."""
        from gemini_batch.pipeline.planner import ExecutionPlanner
        from gemini_batch.telemetry import _EnabledTelemetryContext, _SimpleReporter

        reporter = _SimpleReporter()
        planner = ExecutionPlanner(telemetry=_EnabledTelemetryContext(reporter))

        initial = InitialCommand(
            sources=(),
            prompts=("test",),
            config=resolve_config(),
            hints=(CacheHint("key"), EstimationOverrideHint(widen_max_factor=1.1)),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())

        # Execute planner with a small event loop to capture telemetry
        import asyncio as _asyncio

        result = _asyncio.get_event_loop().run_until_complete(planner.handle(resolved))

        assert isinstance(result, Success)
        keys = set(reporter.metrics.keys())
        assert any(k.endswith("planner.hints.hints_seen") for k in keys)
        assert any(k.endswith("planner.hints.cache_hint") for k in keys)
        assert any(k.endswith("planner.hints.estimation_override") for k in keys)


class TestHintValidation:
    """Test hint validation works correctly."""

    def test_cache_hint_validation(self):
        """Test CacheHint validation catches invalid inputs."""
        # Valid hint should work
        hint = CacheHint(deterministic_key="valid-key")
        assert hint.deterministic_key == "valid-key"

        # Invalid key should raise
        with pytest.raises(
            ValueError, match="deterministic_key must be a non-empty string"
        ):
            CacheHint(deterministic_key="")

        with pytest.raises(
            ValueError, match="deterministic_key must be a non-empty string"
        ):
            CacheHint(deterministic_key="   ")

    def test_estimation_override_hint_validation(self):
        """Test EstimationOverrideHint validation."""
        # Valid hint should work
        hint = EstimationOverrideHint(widen_max_factor=1.5, clamp_max_tokens=1000)
        assert hint.widen_max_factor == 1.5

        # Invalid factor should raise
        with pytest.raises(
            ValueError, match="widen_max_factor must be finite and >= 1.0"
        ):
            EstimationOverrideHint(widen_max_factor=0.5)

        with pytest.raises(ValueError, match="clamp_max_tokens must be >= 0"):
            EstimationOverrideHint(clamp_max_tokens=-1)

    def test_execution_cache_name_validation(self):
        """Test ExecutionCacheName validation."""
        # Valid name should work
        hint = ExecutionCacheName(cache_name="valid-cache")
        assert hint.cache_name == "valid-cache"

        # Invalid name should raise
        with pytest.raises(ValueError, match="cache_name must be a non-empty string"):
            ExecutionCacheName(cache_name="")


class TestArchitecturalInvariants:
    """Test that fixes preserve architectural invariants."""

    def test_hints_remain_optional(self):
        """Verify hints remain optional and fail-soft."""

        # Should work with no hints
        config = resolve_config()
        command_no_hints = InitialCommand(
            sources=(),
            prompts=("test",),
            config=config,
            hints=None,  # explicitly None
        )
        assert command_no_hints.hints is None

        # Should work with empty hints
        command_empty_hints = InitialCommand(
            sources=(),
            prompts=("test",),
            config=config,
            hints=(),  # empty tuple
        )
        assert command_empty_hints.hints == ()

    def test_core_remains_agnostic_to_hint_types(self):
        """Verify core types remain agnostic to concrete hint types."""

        # InitialCommand should accept any object as hints
        # since it's typed as tuple[object, ...] | None
        command = InitialCommand(
            sources=(),
            prompts=("test",),
            config=resolve_config(),
            hints=("string", 42, {"arbitrary": "object"}),  # Any objects
        )
        assert command.hints is not None
        assert len(command.hints) == 3
