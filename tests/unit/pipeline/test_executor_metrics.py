import pytest

from gemini_batch.config import resolve_config
from gemini_batch.core.types import InitialCommand
from gemini_batch.executor import create_executor


@pytest.mark.asyncio
async def test_executor_surfaces_all_stage_durations():
    executor = create_executor(
        resolve_config(programmatic={"api_key": "k"}).to_frozen()
    )
    initial = InitialCommand(
        sources=("s",),
        prompts=("p",),
        config=resolve_config(programmatic={"api_key": "k"}).to_frozen(),
    )

    result = await executor.execute(initial)

    assert result["success"] is True
    metrics = result.get("metrics", {})
    assert isinstance(metrics, dict)
    durations = metrics.get("durations")
    assert isinstance(durations, dict)
    # Expect all stages to be present:
    for stage in ("SourceHandler", "ExecutionPlanner", "APIHandler", "ResultBuilder"):
        assert stage in durations
