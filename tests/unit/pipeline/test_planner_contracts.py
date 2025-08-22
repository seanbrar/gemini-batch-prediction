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
    primary = planned.execution_plan.primary_call
    assert primary.api_parts
    # Joined with blank line between prompts
    first_part = primary.api_parts[0]
    assert hasattr(first_part, "text")
    assert first_part.text == "p1\n\np2"


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
