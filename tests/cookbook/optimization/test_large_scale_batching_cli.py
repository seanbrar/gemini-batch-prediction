from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_large_scale_batching_cli_calls_main_async(
    tmp_path: Path,
    cap_cli: Callable[[Any, list[str], Callable[[], Any]], tuple[str, str]],
) -> None:
    mod: Any = load_recipe_module("cookbook/optimization/large-scale-batching.py")

    d = tmp_path / "data"
    d.mkdir()
    (d / "a.txt").write_text("x")
    (d / "b.txt").write_text("y")
    mod.DEFAULT_TEXT_DEMO_DIR = d

    called: dict[str, Any] = {"dir": None, "prompt": None, "conc": None}

    async def fake_main_async(
        directory: Path,
        prompt: str,
        concurrency: int,
        limit: int = 2,  # noqa: ARG001
    ) -> None:
        called["dir"], called["prompt"], called["conc"] = directory, prompt, concurrency

    mod.main_async = fake_main_async

    out, _ = cap_cli(mod, ["prog", "--prompt", "Q?", "--concurrency", "3"], mod.main)

    assert called["dir"] == d and called["prompt"] == "Q?" and called["conc"] == 3
    assert "Note: File size/count" in out
