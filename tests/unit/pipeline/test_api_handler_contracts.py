import pytest

from gemini_batch.config import GeminiConfig
from gemini_batch.core.types import (
    APICall,
    ExecutionPlan,
    Failure,
    FinalizedCommand,
    InitialCommand,
    PlannedCommand,
    ResolvedCommand,
    Success,
    TextPart,
)
from gemini_batch.pipeline.api_handler import APIHandler


def make_planned(prompts: tuple[str, ...]) -> PlannedCommand:
    initial = InitialCommand(
        sources=("s",), prompts=prompts, config=GeminiConfig(api_key="k")
    )
    resolved = ResolvedCommand(initial=initial, resolved_sources=())
    call = APICall(
        model_name="gemini-2.0-flash",
        api_parts=[TextPart(text="\n\n".join(prompts))],
        api_config={},
    )
    plan = ExecutionPlan(primary_call=call)
    return PlannedCommand(resolved=resolved, execution_plan=plan)


@pytest.mark.asyncio
async def test_api_handler_uses_planned_parts():
    handler = APIHandler()
    planned = make_planned(("hello",))
    result = await handler.handle(planned)
    assert isinstance(result, Success)
    finalized: FinalizedCommand = result.value
    raw = finalized.raw_api_response
    assert isinstance(raw, dict)
    assert "echo: hello" in raw.get("text", "")


@pytest.mark.asyncio
async def test_api_handler_fails_on_empty_parts():
    handler = APIHandler()
    initial = InitialCommand(
        sources=("s",), prompts=("p",), config=GeminiConfig(api_key="k")
    )
    resolved = ResolvedCommand(initial=initial, resolved_sources=())
    empty_call = APICall(model_name="gemini-2.0-flash", api_parts=[], api_config={})
    plan = ExecutionPlan(primary_call=empty_call)
    planned = PlannedCommand(resolved=resolved, execution_plan=plan)

    result = await handler.handle(planned)
    assert isinstance(result, Failure)
