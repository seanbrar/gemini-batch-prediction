import pytest

from gemini_batch.config import resolve_config
from gemini_batch.core.exceptions import PipelineError
from gemini_batch.core.types import InitialCommand
from gemini_batch.executor import create_executor


@pytest.mark.asyncio
async def test_minimal_pipeline_happy_path():
    executor = create_executor(
        resolve_config(
            programmatic={"api_key": "test", "model": "gemini-2.0-flash"}
        ).to_frozen()
    )
    cmd = InitialCommand(
        sources=("hello world",), prompts=("Echo me",), config=executor.config
    )

    result = await executor.execute(cmd)

    assert result["success"] is True
    assert isinstance(result["answers"], list)
    assert result["answers"] and "echo:" in result["answers"][0]


@pytest.mark.asyncio
async def test_pipeline_raises_on_stage_failure():
    executor = create_executor(
        resolve_config(
            programmatic={"api_key": "test", "model": "gemini-2.0-flash"}
        ).to_frozen()
    )
    # Non-existent path should cause SourceHandler to return Failure → executor raises PipelineError
    bad_cmd = InitialCommand(
        sources=("/definitely/not/here.xyz",), prompts=("p",), config=executor.config
    )

    with pytest.raises(PipelineError):
        await executor.execute(bad_cmd)
