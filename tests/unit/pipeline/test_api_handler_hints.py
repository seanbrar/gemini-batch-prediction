from typing import Any

import pytest

from gemini_batch.config import resolve_config
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
from gemini_batch.pipeline.hints import CacheHint, ExecutionCacheName
from gemini_batch.pipeline.registries import CacheRegistry

pytestmark = pytest.mark.unit


class _HintsSpyAdapter:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.hints: list[Any] = []
        self.first = True

    # ExecutionHintsAware-compatible
    def apply_hints(self, hints: Any) -> None:
        self.hints.append(hints)

    async def create_cache(
        self,
        *,
        model_name: str,  # noqa: ARG002
        content_parts: tuple[Any, ...],  # noqa: ARG002
        system_instruction: str | None,  # noqa: ARG002
        ttl_seconds: int | None,  # noqa: ARG002
    ) -> str:
        return "cachedContents/fake-123"

    async def generate(
        self,
        *,
        model_name: str,  # noqa: ARG002
        api_parts: tuple[Any, ...],  # noqa: ARG002
        api_config: dict[str, object],
    ) -> dict[str, Any]:
        # Fail on first call if cached_content present; succeed thereafter
        if self.first and "cached_content" in api_config:
            self.first = False
            raise RuntimeError("simulate cache-too-large")
        return {
            "text": "ok",
            "model": "gemini-2.0-flash",
            "usage": {"total_token_count": 5},
        }


class TestAPIHandlerHints:
    """Test API handler hint consumption and retry semantics."""

    @pytest.mark.asyncio
    async def test_execution_hints_applied_and_retry_without_cache(self):
        """Explicit cache plan should trigger retry without cache on failure."""
        # Plan with explicit cache creation to trigger cached_content hint
        initial = InitialCommand(
            sources=("s",),
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

        spy = _HintsSpyAdapter()
        handler = APIHandler(adapter=spy)
        result = await handler.handle(planned)
        assert isinstance(result, Success)
        # Hints should have been applied at least twice: with cache then without
        cached_values = [getattr(h, "cached_content", None) for h in spy.hints]
        assert any(isinstance(v, str) for v in cached_values)  # with cache
        assert any(v is None for v in cached_values)  # retried without cache

    @pytest.mark.asyncio
    async def test_execution_cache_name_hint_overrides_cache_name(self):
        """ExecutionCacheName hint should override cache name."""
        initial = InitialCommand(
            sources=(),
            prompts=("test",),
            config=resolve_config(overrides={"api_key": "k"}),
            hints=(ExecutionCacheName("override-cache"),),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())
        call = APICall(
            model_name="gemini-2.0-flash",
            api_parts=(TextPart("test"),),
            api_config={},
            cache_name_to_use="original-cache",
        )
        plan = ExecutionPlan(primary_call=call)
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        spy = _HintsSpyAdapter()
        handler = APIHandler(adapter=spy)
        result = await handler.handle(planned)

        assert isinstance(result, Success)
        # Should have used the override cache name
        cached_values = [getattr(h, "cached_content", None) for h in spy.hints]
        assert "override-cache" in cached_values

    @pytest.mark.asyncio
    async def test_execution_cache_name_hint_triggers_retry_on_failure(self):
        """ExecutionCacheName hint should trigger retry without cache on failure."""
        initial = InitialCommand(
            sources=(),
            prompts=("test",),
            config=resolve_config(overrides={"api_key": "k"}),
            hints=(ExecutionCacheName("failing-cache"),),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())
        call = APICall(
            model_name="gemini-2.0-flash", api_parts=(TextPart("test"),), api_config={}
        )
        plan = ExecutionPlan(primary_call=call)
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        spy = _HintsSpyAdapter()
        handler = APIHandler(adapter=spy)
        result = await handler.handle(planned)

        assert isinstance(result, Success)
        # Should have tried with override cache, then retried without
        cached_values = [getattr(h, "cached_content", None) for h in spy.hints]
        assert "failing-cache" in cached_values  # First attempt with hint override
        assert None in cached_values  # Retry without cache

    @pytest.mark.asyncio
    async def test_no_execution_cache_hint_preserves_original_behavior(self):
        """Without ExecutionCacheName hint, original cache behavior preserved."""
        initial = InitialCommand(
            sources=(),
            prompts=("test",),
            config=resolve_config(overrides={"api_key": "k"}),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())
        call = APICall(
            model_name="gemini-2.0-flash",
            api_parts=(TextPart("test"),),
            api_config={},
            cache_name_to_use="original-cache",
        )
        plan = ExecutionPlan(primary_call=call)
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        # Use adapter that doesn't fail on cache
        class _SuccessAdapter:
            def apply_hints(self, hints: Any) -> None:
                pass

            async def generate(
                self,
                *,
                model_name: str,
                api_parts: tuple[Any, ...],  # noqa: ARG002
                api_config: dict[str, object],  # noqa: ARG002
            ) -> dict[str, Any]:
                return {
                    "text": "success",
                    "model": model_name,
                    "usage": {"total_token_count": 5},
                }

        handler = APIHandler(adapter=_SuccessAdapter())
        result = await handler.handle(planned)

        assert isinstance(result, Success)
        # Should have succeeded without hints

    @pytest.mark.asyncio
    async def test_malformed_hints_handled_gracefully(self):
        """Malformed hints should be handled gracefully without failing execution."""
        # Create initial command with malformed hints
        initial = InitialCommand(
            sources=(),
            prompts=("test",),
            config=resolve_config(overrides={"api_key": "k"}),
            hints=("not-a-tuple",),  # Malformed hint as tuple
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())
        call = APICall(
            model_name="gemini-2.0-flash", api_parts=(TextPart("test"),), api_config={}
        )
        plan = ExecutionPlan(primary_call=call)
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        spy = _HintsSpyAdapter()
        handler = APIHandler(adapter=spy)
        result = await handler.handle(planned)

        # Should still succeed despite malformed hints
        assert isinstance(result, Success)

    @pytest.mark.asyncio
    async def test_no_retry_without_caching_capability_even_with_explicit_plan(self):
        """Adapter lacking caching capability should not trigger special no-cache retry."""
        from gemini_batch.telemetry import _EnabledTelemetryContext, _SimpleReporter

        class _NoCacheAdapter:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []
                self.first = True

            async def generate(
                self,
                *,
                model_name: str,
                api_parts: tuple[Any, ...],  # noqa: ARG002
                api_config: dict[str, object],
            ) -> dict[str, object]:
                # Simulate transient failure once, then success (no cached_content key used)
                self.calls.append(dict(api_config))
                if self.first:
                    self.first = False
                    raise RuntimeError("timeout while calling provider")
                return {
                    "text": "ok",
                    "model": model_name,
                    "usage": {"total_token_count": 2},
                }

        reporter = _SimpleReporter()
        handler = APIHandler(
            adapter=_NoCacheAdapter(), telemetry=_EnabledTelemetryContext(reporter)
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
        # Include an explicit cache plan, but adapter lacks CachingCapability so no cache is applied
        plan = ExecutionPlan(
            primary_call=call,
            explicit_cache=ExplicitCachePlan(create=True, contents_part_indexes=(0,)),
        )
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        result = await handler.handle(planned)
        assert isinstance(result, Success)
        # Two generate calls: initial + transient backoff retry; no special no-cache retry scope
        adapter = handler._adapter
        assert adapter is not None
        assert isinstance(adapter, _NoCacheAdapter)
        assert len(adapter.calls) == 2
        assert all("cached_content" not in c for c in adapter.calls)
        assert not any(k.endswith("api.retry_no_cache") for k in reporter.timings)

    @pytest.mark.asyncio
    async def test_explicit_cache_name_with_create_false_used_and_retry_on_failure(
        self,
    ):
        """ExplicitCachePlan with cache_name (create=False) should be used and retried once without cache on failure."""
        from gemini_batch.telemetry import _EnabledTelemetryContext, _SimpleReporter

        class _CapableAdapter:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []
                self.first = True

            async def create_cache(
                self,
                *,
                model_name: str,  # noqa: ARG002
                content_parts: tuple[Any, ...],  # noqa: ARG002
                system_instruction: str | None,  # noqa: ARG002
                ttl_seconds: int | None,  # noqa: ARG002
            ) -> str:
                return "cachedContents/unused"

            async def generate(
                self,
                *,
                model_name: str,
                api_parts: tuple[Any, ...],  # noqa: ARG002
                api_config: dict[str, object],
            ) -> dict[str, object]:
                self.calls.append(dict(api_config))
                if self.first and "cached_content" in api_config:
                    self.first = False
                    raise RuntimeError("fail with cache")
                return {
                    "text": "ok",
                    "model": model_name,
                    "usage": {"total_token_count": 2},
                }

        reporter = _SimpleReporter()
        handler = APIHandler(
            adapter=_CapableAdapter(), telemetry=_EnabledTelemetryContext(reporter)
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
            explicit_cache=ExplicitCachePlan(
                create=False,
                cache_name="prebuilt-cache",
                contents_part_indexes=(0,),
                deterministic_key="dk",
            ),
        )
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        result = await handler.handle(planned)
        assert isinstance(result, Success)
        # First with cache, then retry without cache
        adapter = handler._adapter
        assert adapter is not None
        assert isinstance(adapter, _CapableAdapter)
        flags = ["cached_content" in c for c in adapter.calls]
        assert flags == [True, False]

    @pytest.mark.asyncio
    async def test_override_precedence_does_not_mutate_plan(self):
        """ExecutionCacheName override should win at execution time but not change the plan."""

        class _SpyHints:
            def __init__(self) -> None:
                self.hints_seen: list[Any] = []

            def apply_hints(self, hints: Any) -> None:
                self.hints_seen.append(hints)

            async def generate(
                self,
                *,
                model_name: str,
                api_parts: tuple[Any, ...],  # noqa: ARG002
                api_config: dict[str, object],  # noqa: ARG002
            ) -> dict[str, object]:
                return {
                    "text": "ok",
                    "model": model_name,
                    "usage": {"total_token_count": 1},
                }

        spy = _SpyHints()
        handler = APIHandler(adapter=spy)

        initial = InitialCommand(
            sources=(),
            prompts=("p",),
            config=resolve_config(overrides={"api_key": "k"}),
            hints=(ExecutionCacheName("override-cache"),),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())
        call = APICall(
            model_name="gemini-2.0-flash",
            api_parts=(TextPart("p"),),
            api_config={},
            cache_name_to_use="plan-cache",
        )
        plan = ExecutionPlan(primary_call=call)
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        result = await handler.handle(planned)
        assert isinstance(result, Success)
        # Adapter should have seen override cache in hints; plan remains unchanged
        cached_values = [getattr(h, "cached_content", None) for h in spy.hints_seen]
        assert "override-cache" in cached_values
        assert planned.execution_plan.primary_call.cache_name_to_use == "plan-cache"

    @pytest.mark.asyncio
    async def test_cache_hint_artifacts_written_to_registry_best_effort(self):
        """Artifacts from CacheHint should be recorded into registry metadata when possible."""

        # Adapter capable of caching; we provide a fixed cache_name to avoid create call
        class _Adapter:
            async def generate(
                self,
                *,
                model_name: str,
                api_parts: tuple[Any, ...],  # noqa: ARG002
                api_config: dict[str, object],  # noqa: ARG002
            ) -> dict[str, object]:
                return {
                    "text": "ok",
                    "model": model_name,
                    "usage": {"total_token_count": 1},
                }

            async def create_cache(
                self,
                *,
                model_name: str,  # noqa: ARG002
                content_parts: tuple[Any, ...],  # noqa: ARG002
                system_instruction: str | None,  # noqa: ARG002
                ttl_seconds: int | None,  # noqa: ARG002
            ) -> str:
                return "cachedContents/ignored"

        reg = CacheRegistry()
        handler = APIHandler(adapter=_Adapter(), registries={"cache": reg})

        initial = InitialCommand(
            sources=(),
            prompts=("p",),
            config=resolve_config(overrides={"api_key": "k"}),
            hints=(CacheHint("det-key", artifacts=("one", "two")),),
        )
        resolved = ResolvedCommand(initial=initial, resolved_sources=())
        call = APICall(
            model_name="gemini-2.0-flash", api_parts=(TextPart("p"),), api_config={}
        )
        plan = ExecutionPlan(
            primary_call=call,
            explicit_cache=ExplicitCachePlan(
                create=False,
                cache_name="cn",
                contents_part_indexes=(0,),
                deterministic_key="det-key",
            ),
        )
        planned = PlannedCommand(resolved=resolved, execution_plan=plan)

        result = await handler.handle(planned)
        assert isinstance(result, Success)
        meta = reg.get_meta("det-key")
        assert isinstance(meta, dict)
        assert meta.get("cache_name") == "cn"
        assert tuple(meta.get("artifacts", ())) == ("one", "two")
