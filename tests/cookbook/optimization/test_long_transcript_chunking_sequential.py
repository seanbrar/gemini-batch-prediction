from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.cookbook.support import load_recipe_module


@pytest.mark.unit
@pytest.mark.cookbook
def test_long_transcript_chunking_sequential_uses_run_batch(tmp_path, capsys):
    mod: Any = load_recipe_module("cookbook/optimization/long-transcript-chunking.py")

    f = tmp_path / "t.txt"
    f.write_text("A\nB\nC\n")

    # Force chunker to 2 chunks
    chunks = [
        mod.TranscriptChunk(0.0, 10.0, "X", (mod.TranscriptSegment(0.0, 10.0, "X"),)),
        mod.TranscriptChunk(10.0, 20.0, "Y", (mod.TranscriptSegment(10.0, 20.0, "Y"),)),
    ]

    mod.chunk_transcript_by_tokens = lambda *_args, **_kw: chunks

    calls = {"n": 0}

    async def fake_run_batch(_prompts, _sources):
        calls["n"] += 1
        return {"answers": [f"ans{calls['n']}"]}

    mod.run_batch = fake_run_batch
    # Ensure parallel path is not taken
    mod.run_parallel = None

    asyncio.run(mod.main_async(f, target_tokens=500, prompt="P", concurrency=1))
    out = capsys.readouterr().out
    assert "Created 2 chunks" in out and "Consolidated answer" in out
    assert calls["n"] == 2
