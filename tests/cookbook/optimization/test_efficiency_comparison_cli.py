from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_efficiency_comparison_cli_calls_main_async(
    tmp_path: Path,
    cap_cli: Callable[[Any, list[str], Callable[[], Any]], tuple[str, str]],
) -> None:
    mod: Any = load_recipe_module("cookbook/optimization/efficiency-comparison.py")
    called: dict[str, Any] = {"dir": None, "mode": None, "trials": None}

    async def fake_main_async(
        directory: Path,
        mode: str,
        trials: int,
        limit: int = 2,  # noqa: ARG001
    ) -> None:
        called["dir"] = directory
        called["mode"] = mode
        called["trials"] = trials

    mod.main_async = fake_main_async

    d = tmp_path
    (d / "a.txt").write_text("x")
    argv = ["prog", "--input", str(d), "--mode", "aggregate", "--trials", "3"]
    out, _ = cap_cli(mod, argv, mod.main)

    assert called["dir"] == d
    assert called["mode"] == "aggregate"
    assert called["trials"] == 3
    assert "Note: File size/count affect runtime and tokens." in out
