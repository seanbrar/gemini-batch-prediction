import pytest

from gemini_batch.config import GeminiConfig
from gemini_batch.core.types import InitialCommand, ResolvedCommand, Source, Success
from gemini_batch.pipeline.planner import ExecutionPlanner

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_planner_includes_prompt_in_token_estimate():
    planner = ExecutionPlanner()

    # Prompt-only scenario
    initial = InitialCommand(
        sources=(),
        prompts=("short prompt",),
        config=GeminiConfig(api_key="k", model="gemini-2.0-flash"),
    )
    resolved = ResolvedCommand(initial=initial, resolved_sources=())

    result = await planner.handle(resolved)
    assert isinstance(result, Success)
    planned = result.value

    # Token estimate should be present and include the prompt breakdown
    assert planned.token_estimate is not None
    estimate = planned.token_estimate
    assert estimate.expected_tokens >= 10  # prompt-only should be at least the floor
    assert estimate.breakdown is None or "prompt" in estimate.breakdown

    # Parts should contain the joined prompt
    primary = planned.execution_plan.primary_call
    assert primary.api_parts
    assert primary.api_parts[0].text == "short prompt"


@pytest.mark.asyncio
async def test_cache_key_is_deterministic_and_changes_with_prompts(monkeypatch):
    planner = ExecutionPlanner()

    # Force caching decision for this test
    monkeypatch.setattr(planner, "_should_cache", lambda *args, **kwargs: True)  # noqa: ARG005

    # Build a large source to be part of the cache key payload
    large_source = Source(
        source_type="file",
        identifier="/dev/null",
        mime_type="application/octet-stream",
        size_bytes=10_000_000,
        content_loader=lambda: b"",
    )

    initial_a = InitialCommand(
        sources=("ignored",),
        prompts=("A",),
        config=GeminiConfig(api_key="k", model="gemini-2.0-flash"),
    )
    resolved_a = ResolvedCommand(initial=initial_a, resolved_sources=(large_source,))

    initial_b = InitialCommand(
        sources=("ignored",),
        prompts=("A",),
        config=GeminiConfig(api_key="k", model="gemini-2.0-flash"),
    )
    resolved_b = ResolvedCommand(initial=initial_b, resolved_sources=(large_source,))
    # Deterministic: identical inputs produce identical cache names
    result_a = await planner.handle(resolved_a)
    assert isinstance(result_a, Success)
    planned_a = result_a.value

    result_b = await planner.handle(resolved_b)
    assert isinstance(result_b, Success)
    planned_b = result_b.value

    cache_a = planned_a.execution_plan.primary_call.cache_name_to_use
    cache_b = planned_b.execution_plan.primary_call.cache_name_to_use
    assert cache_a and cache_b and cache_a == cache_b

    # Changing prompts should yield a different cache name
    initial_c = InitialCommand(
        sources=("ignored",),
        prompts=("B",),
        config=GeminiConfig(api_key="k", model="gemini-2.0-flash"),
    )
    resolved_c = ResolvedCommand(initial=initial_c, resolved_sources=(large_source,))
    result_c = await planner.handle(resolved_c)
    assert isinstance(result_c, Success)
    planned_c = result_c.value
    cache_c = planned_c.execution_plan.primary_call.cache_name_to_use
    assert cache_c and cache_c != cache_a
