"""Behavioral tests for SourceHandler source resolution.

These tests verify that legacy file/url/text handling is preserved while
producing the new `Source` dataclass outputs expected by the pipeline.
"""

from pathlib import Path

import pytest

from gemini_batch.config import GeminiConfig
from gemini_batch.core.exceptions import SourceError
from gemini_batch.core.types import Failure, InitialCommand, Success
from gemini_batch.pipeline.source_handler import SourceHandler


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.mark.asyncio
async def test_resolves_text_content():
    handler = SourceHandler()
    text = "Hello world"
    command = InitialCommand(
        sources=(text,), prompts=("p",), config=GeminiConfig(api_key="test")
    )

    result = await handler.handle(command)
    assert isinstance(result, Success)
    resolved = result.value
    assert len(resolved.resolved_sources) == 1
    src = resolved.resolved_sources[0]
    assert src.source_type == "text"
    assert src.identifier == text
    assert src.mime_type == "text/plain"
    assert src.content_loader() == text.encode()


@pytest.mark.asyncio
async def test_resolves_file_path_text_file():
    handler = SourceHandler()
    file_path = project_root() / "test_files" / "report.txt"
    assert file_path.exists(), f"Missing test fixture: {file_path}"

    command = InitialCommand(
        sources=(str(file_path),), prompts=("p",), config=GeminiConfig(api_key="test")
    )
    result = await handler.handle(command)
    assert isinstance(result, Success)
    src = result.value.resolved_sources[0]
    assert src.source_type == "file"
    assert Path(src.identifier) == file_path
    assert src.content_loader() == file_path.read_bytes()


@pytest.mark.asyncio
async def test_resolves_directory_expansion():
    handler = SourceHandler()
    dir_path = project_root() / "test_files"
    assert dir_path.exists() and dir_path.is_dir(), (
        f"Missing test fixture directory: {dir_path}"
    )

    command = InitialCommand(
        sources=(str(dir_path),), prompts=("p",), config=GeminiConfig(api_key="test")
    )
    result = await handler.handle(command)
    assert isinstance(result, Success)
    sources = result.value.resolved_sources
    # Expect multiple sources from directory contents
    assert len(sources) >= 3
    # All should be file sources
    assert all(s.source_type == "file" for s in sources)


@pytest.mark.asyncio
async def test_resolves_youtube_url():
    handler = SourceHandler()
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    command = InitialCommand(
        sources=(url,), prompts=("p",), config=GeminiConfig(api_key="test")
    )
    result = await handler.handle(command)
    assert isinstance(result, Success)
    src = result.value.resolved_sources[0]
    assert src.source_type == "youtube"
    assert src.identifier == url
    assert src.mime_type == "video/youtube"


@pytest.mark.asyncio
async def test_resolves_arxiv_pdf_url():
    handler = SourceHandler()
    url = "https://arxiv.org/pdf/1706.03762.pdf"
    command = InitialCommand(
        sources=(url,), prompts=("p",), config=GeminiConfig(api_key="test")
    )
    result = await handler.handle(command)
    assert isinstance(result, Success)
    src = result.value.resolved_sources[0]
    assert src.source_type == "arxiv"
    assert src.identifier == url
    assert src.mime_type == "application/pdf"


@pytest.mark.asyncio
async def test_error_on_nonexistent_path():
    handler = SourceHandler()
    missing = project_root() / "test_files" / "does_not_exist.xyz"
    command = InitialCommand(
        sources=(str(missing),), prompts=("p",), config=GeminiConfig(api_key="test")
    )
    result = await handler.handle(command)
    # Should fail explicitly, not raise

    assert isinstance(result, Failure)
    assert isinstance(result.error, SourceError)
