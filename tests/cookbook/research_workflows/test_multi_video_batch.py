from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.cookbook.support import load_recipe_module


@pytest.mark.unit
@pytest.mark.cookbook
def test_multi_video_batch_prints_status_and_per_prompt(tmp_path, capsys):
    # Arrange: combine a URL and a local file
    local = tmp_path / "clip.mov"
    local.write_bytes(b"\x00\x00")
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    mod: Any = load_recipe_module("cookbook/research-workflows/multi-video-batch.py")
    main_async = mod.main_async

    fake_env = {
        "status": "ok",
        "answers": ["a", "b", "c"],
        "usage": {"total_token_count": 1234},
        "metrics": {
            "per_prompt": [
                {"index": 0, "durations": {"execute.total": 0.123}},
                {"index": 1, "durations": {"execute.total": 0.234}},
                {"index": 2, "durations": {"execute.total": 0.345}},
            ]
        },
    }

    async def fake_run_batch(*_args, **_kwargs):
        return fake_env

    mod.run_batch = fake_run_batch

    # Act
    asyncio.run(main_async([url, local]))

    # Assert
    out = capsys.readouterr().out
    assert "Status: ok" in out
    assert "Answers: 3 prompts returned" in out
    assert "Total tokens:" in out
    assert "Per-prompt snapshots" in out
    assert "Prompt[0] duration:" in out
