import pytest

from gemini_batch.config import GeminiConfig
from gemini_batch.core.types import InitialCommand
from gemini_batch.executor import create_executor


@pytest.mark.asyncio
async def test_executor_surfaces_all_stage_durations():
    executor = create_executor(GeminiConfig(api_key="k", model="gemini-2.0-flash"))
    cmd = InitialCommand(sources=("s",), prompts=("p",), config=executor.config)

    result = await executor.execute(cmd)

    assert result["success"] is True
    metrics = result.get("metrics", {})
    assert isinstance(metrics, dict)
    durations = metrics.get("durations")
    assert isinstance(durations, dict)
    # Expect all stages to be present:
    for stage in ("SourceHandler", "ExecutionPlanner", "APIHandler", "ResultBuilder"):
        assert stage in durations
