from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_multi_video_batch_cli_calls_main_async(
    tmp_path: Path,
    cap_cli: Callable[[Any, list[str], Callable[[], Any]], tuple[str, str]],
) -> None:
    mod: Any = load_recipe_module("cookbook/research-workflows/multi-video-batch.py")

    local = tmp_path / "clip.mp4"
    local.write_bytes(b"\x00")
    url = "https://example.com/watch?v=abc"

    called = {"inputs": None}

    async def fake_main_async(inputs):
        called["inputs"] = inputs

    mod.main_async = fake_main_async

    cap_cli(mod, ["prog", url, str(local)], mod.main)

    # Inputs are normalized: url stays str, path becomes Path
    assert isinstance(called["inputs"], list)
    assert url in called["inputs"] and local in called["inputs"]
