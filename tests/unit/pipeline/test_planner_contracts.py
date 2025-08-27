import pytest

from gemini_batch.config import resolve_config
from gemini_batch.core.types import (
    Failure,
    InitialCommand,
    ResolvedCommand,
    Success,
)
from gemini_batch.pipeline.planner import ExecutionPlanner

pytestmark = pytest.mark.contract


@pytest.mark.asyncio
async def test_planner_produces_parts_on_prompts():
    planner = ExecutionPlanner()
    initial = InitialCommand(
        sources=("src",),
        prompts=("p1", "p2"),
        config=resolve_config(overrides={"api_key": "k"}),
    )
    resolved = ResolvedCommand(initial=initial, resolved_sources=())

    result = await planner.handle(resolved)

    assert isinstance(result, Success)
    planned = result.value
    plan = planned.execution_plan
    # Vectorized path: two independent calls, primary_call mirrors the first call
    assert plan.calls and len(plan.calls) == 2
    first_call = plan.calls[0]
    assert first_call.api_parts and hasattr(first_call.api_parts[0], "text")
    assert getattr(first_call.api_parts[0], "text") == "p1"
    # primary_call is kept for back-compat but should match the first call
    assert plan.primary_call.model_name == first_call.model_name
    assert plan.primary_call.api_config == first_call.api_config
    assert plan.primary_call.api_parts == first_call.api_parts


@pytest.mark.asyncio
async def test_planner_fails_on_empty_prompts():
    planner = ExecutionPlanner()
    initial = InitialCommand(
        sources=("src",),
        prompts=(),
        config=resolve_config(overrides={"api_key": "k"}),
    )
    resolved = ResolvedCommand(initial=initial, resolved_sources=())

    result = await planner.handle(resolved)

    assert isinstance(result, Failure)
