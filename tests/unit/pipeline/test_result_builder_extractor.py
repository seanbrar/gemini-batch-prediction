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
from gemini_batch.pipeline.result_builder import ResultBuilder


def make_finalized_with_raw(raw: object) -> FinalizedCommand:
    """Create a minimal valid FinalizedCommand for testing."""
    initial = InitialCommand(
        sources=("s",), prompts=("p",), config=GeminiConfig(api_key="k")
    )
    resolved = ResolvedCommand(initial=initial, resolved_sources=())
    api_call = APICall(
        model_name="gemini-2.0-flash", api_parts=[TextPart(text="p")], api_config={}
    )
    plan = ExecutionPlan(primary_call=api_call)
    planned = PlannedCommand(resolved=resolved, execution_plan=plan)
    return FinalizedCommand(planned=planned, raw_api_response=raw)


@pytest.mark.asyncio
async def test_extractor_accepts_simple_text():
    handler = ResultBuilder()
    cmd = make_finalized_with_raw({"text": "hello"})
    result = await handler.handle(cmd)
    assert isinstance(result, Success)
    assert result.value["answers"][0] == "hello"


@pytest.mark.asyncio
async def test_extractor_accepts_nested_shape():
    handler = ResultBuilder()
    raw = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "nested"},
                    ]
                }
            }
        ]
    }
    cmd = make_finalized_with_raw(raw)
    result = await handler.handle(cmd)
    assert isinstance(result, Success)
    assert result.value["answers"][0] == "nested"


@pytest.mark.asyncio
async def test_extractor_fails_when_text_missing():
    handler = ResultBuilder()
    cmd = make_finalized_with_raw({"nope": ""})
    result = await handler.handle(cmd)
    assert isinstance(result, Failure)
