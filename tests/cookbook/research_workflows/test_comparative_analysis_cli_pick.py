from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_comparative_analysis_cli_picks_two_from_directory(
    tmp_path: Path,
    cap_cli: Callable[[Any, list[str], Callable[[], Any]], tuple[str, str]],
) -> None:
    mod: Any = load_recipe_module("cookbook/research-workflows/comparative-analysis.py")
    # Create two files so CLI will auto-pick them
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")

    payload = {
        "similarities": ["s1"],
        "differences": ["d1"],
        "strengths": [],
        "weaknesses": [],
    }

    async def fake_run_batch(*_args, **_kwargs):
        return {"answers": [json.dumps(payload)]}

    mod.run_batch = fake_run_batch

    out, _ = cap_cli(mod, ["prog", "--input", str(tmp_path)], mod.main)

    assert "Comparison Summary" in out and "Similarities: 1" in out
