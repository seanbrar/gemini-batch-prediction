from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_long_transcript_chunking_creates_chunks_and_stitches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    patch_run_batch: Callable[..., None],
    patch_run_parallel: Callable[..., None],
    make_env: Callable[..., dict[str, Any]],
) -> None:
    # Arrange transcript with multiple lines (content irrelevant when chunking is patched)
    path = tmp_path / "transcript.txt"
    path.write_text("Line 1\nLine 2\nLine 3\nLine 4\n")

    mod: Any = load_recipe_module("cookbook/optimization/long-transcript-chunking.py")
    main_async = mod.main_async

    # Patch chunking to return exactly 3 chunks with simple text
    chunks = [
        mod.TranscriptChunk(
            0.0,
            10.0,
            "A",
            (mod.TranscriptSegment(0.0, 10.0, "A"),),
        ),
        mod.TranscriptChunk(
            10.0,
            20.0,
            "B",
            (mod.TranscriptSegment(10.0, 20.0, "B"),),
        ),
        mod.TranscriptChunk(
            20.0,
            30.0,
            "C",
            (mod.TranscriptSegment(20.0, 30.0, "C"),),
        ),
    ]

    def fake_chunker(_segs: Any, **_kwargs: Any) -> list[Any]:
        return chunks

    mod.chunk_transcript_by_tokens = fake_chunker

    async def fake_run_parallel(_prompt, sources, **_kwargs):
        # Return one answer per source, in order
        return make_env(answers=[f"ans:{i}" for i, _ in enumerate(sources)])

    async def fake_run_batch(_prompts, _sources):
        return make_env(answers=["ans:0"])

    patch_run_parallel(mod, fake_run_parallel)
    patch_run_batch(mod, fake_run_batch)

    # Act
    asyncio.run(main_async(path, target_tokens=500, prompt="P", concurrency=2))

    # Assert
    out = capsys.readouterr().out
    assert "Created 3 chunks" in out
    assert "Consolidated answer" in out
