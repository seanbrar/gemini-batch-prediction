from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_rate_limits_cli_calls_main_async(
    tmp_path: Path,
    cap_cli: Callable[[Any, list[str], Callable[[], Any]], tuple[str, str]],
) -> None:
    mod: Any = load_recipe_module("cookbook/production/rate-limits-and-concurrency.py")

    d = tmp_path / "data"
    d.mkdir()
    (d / "a.txt").write_text("x")
    mod.DEFAULT_TEXT_DEMO_DIR = d

    called: dict[str, Any] = {"dir": None, "conc": None}

    async def fake_main_async(
        directory: Path,
        concurrency: int,
        limit: int = 2,  # noqa: ARG001
    ) -> None:
        called["dir"], called["conc"] = directory, concurrency

    mod.main_async = fake_main_async

    out, _ = cap_cli(mod, ["prog", "--concurrency", "7"], mod.main)

    assert called["dir"] == d and called["conc"] == 7
    assert "Note: File size/count" in out
