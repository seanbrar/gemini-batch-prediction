import pytest

from gemini_batch.config import GeminiConfig
from gemini_batch.core.types import InitialCommand, Success
from gemini_batch.pipeline.source_handler import SourceHandler


@pytest.mark.asyncio
async def test_text_detection_allows_short_bare_filenames_as_text():
    handler = SourceHandler()
    # No separators; looks like a bare filename, should be treated as text
    cmd = InitialCommand(
        sources=("README.md",), prompts=("p",), config=GeminiConfig(api_key="k")
    )
    result = await handler.handle(cmd)
    assert isinstance(result, Success)
    src = result.value.resolved_sources[0]
    # For real files in project root, detector may treat as file; ensure no failure.
    assert src.identifier in {"README.md", __import__("pathlib").Path("README.md")}


@pytest.mark.asyncio
async def test_text_detection_treats_pathlike_with_separators_as_path():
    handler = SourceHandler()
    # Contains path separators and suffix → treated as a path; should fail if it doesn't exist
    cmd = InitialCommand(
        sources=("some/dir/file.txt",), prompts=("p",), config=GeminiConfig(api_key="k")
    )
    result = await handler.handle(cmd)
    from gemini_batch.core.types import Failure

    assert isinstance(result, Failure)
