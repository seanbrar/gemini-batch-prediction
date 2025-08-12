import pytest

from gemini_batch.config import GeminiConfig
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
        config=GeminiConfig(api_key="k", model="gemini-2.0-flash"),
    )
    resolved = ResolvedCommand(initial=initial, resolved_sources=())

    result = await planner.handle(resolved)

    assert isinstance(result, Success)
    planned = result.value
    primary = planned.execution_plan.primary_call
    assert primary.api_parts
    # Joined with blank line between prompts
    assert primary.api_parts[0].text == "p1\n\np2"


@pytest.mark.asyncio
async def test_planner_fails_on_empty_prompts():
    planner = ExecutionPlanner()
    initial = InitialCommand(
        sources=("src",),
        prompts=(),
        config=GeminiConfig(api_key="k", model="gemini-2.0-flash"),
    )
    resolved = ResolvedCommand(initial=initial, resolved_sources=())

    result = await planner.handle(resolved)

    assert isinstance(result, Failure)
