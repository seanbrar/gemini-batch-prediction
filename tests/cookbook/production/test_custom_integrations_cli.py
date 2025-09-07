from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_custom_integrations_cli_calls_main_async(
    tmp_path: Path,
    cap_cli: Callable[[Any, list[str], Callable[[], Any]], tuple[str, str]],
) -> None:
    mod: Any = load_recipe_module("cookbook/production/custom-integrations.py")

    d = tmp_path / "data"
    d.mkdir()
    (d / "a.txt").write_text("x")
    mod.DEFAULT_TEXT_DEMO_DIR = d

    called: dict[str, Path | None] = {"dir": None}

    async def fake_main_async(directory: Path, limit: int = 2) -> None:  # noqa: ARG001
        called["dir"] = directory

    mod.main_async = fake_main_async
    cap_cli(mod, ["prog"], mod.main)

    assert called["dir"] == d
