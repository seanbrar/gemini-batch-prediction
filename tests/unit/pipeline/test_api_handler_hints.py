from typing import Any

import pytest

from gemini_batch.config import GeminiConfig
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


@pytest.mark.asyncio
async def test_execution_hints_applied_and_retry_without_cache():
    # Plan with explicit cache creation to trigger cached_content hint
    initial = InitialCommand(
        sources=("s",), prompts=("p",), config=GeminiConfig(api_key="k")
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
