from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_long_transcript_chunking_cli_calls_main_async(
    tmp_path: Path,
    cap_cli: Callable[[Any, list[str], Callable[[], Any]], tuple[str, str]],
) -> None:
    mod: Any = load_recipe_module("cookbook/optimization/long-transcript-chunking.py")

    # Create a transcript file and call CLI with positional path
    f = tmp_path / "t.txt"
    f.write_text("a\nb\n")

    called: dict[str, Any] = {"path": None, "tok": None, "prompt": None, "conc": None}

    async def fake_main_async(
        path: Path, target_tokens: int, prompt: str, concurrency: int
    ) -> None:
        called["path"], called["tok"], called["prompt"], called["conc"] = (
            path,
            target_tokens,
            prompt,
            concurrency,
        )

    mod.main_async = fake_main_async

    argv = [
        "prog",
        str(f),
        "--target-tokens",
        "500",
        "--concurrency",
        "2",
        "--prompt",
        "P",
    ]
    cap_cli(mod, argv, mod.main)

    assert (
        called["path"] == f
        and called["tok"] == 500
        and called["conc"] == 2
        and called["prompt"] == "P"
    )
