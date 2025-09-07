from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_system_instructions_cli_calls_run_with_builder(
    tmp_path: Path,
    cap_cli: Callable[[Any, list[str], Callable[[], Any]], tuple[str, str]],
) -> None:
    mod: Any = load_recipe_module(
        "cookbook/research-workflows/system-instructions-with-research-helper.py"
    )

    d = tmp_path / "data"
    d.mkdir()
    (d / "a.txt").write_text("x")
    mod.DEFAULT_TEXT_DEMO_DIR = d

    called: dict[str, Any] = {
        "dir": None,
        "mode": None,
        "trials": None,
        "builder": None,
    }

    async def fake_run_with_system(
        directory: Path, *, mode: str, trials: int, builder: Any, **_kwargs: Any
    ) -> None:
        called.update(
            {"dir": directory, "mode": mode, "trials": trials, "builder": builder}
        )

    mod.run_with_system = fake_run_with_system

    argv = ["prog", "--mode", "aggregate", "--trials", "2", "--use-default-builder"]
    cap_cli(mod, argv, mod.main)

    assert (
        called["dir"] == d and called["mode"] == "aggregate" and called["trials"] == 2
    )
    assert called["builder"] is not None
