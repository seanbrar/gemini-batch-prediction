from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_comparative_analysis_prints_counts_and_first_difference(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    patch_run_batch: Callable[..., None],
    make_env: Callable[..., dict[str, Any]],
) -> None:
    # Arrange two files
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("aaa")
    b.write_text("bbb")

    mod: Any = load_recipe_module("cookbook/research-workflows/comparative-analysis.py")
    main_async = mod.main_async

    payload = {
        "similarities": ["sim1", "sim2"],
        "differences": ["diff1"],
        "strengths": ["s1"],
        "weaknesses": ["w1", "w2"],
    }

    async def fake_run_batch(*_args, **_kwargs):
        return make_env(answers=[json.dumps(payload)])

    patch_run_batch(mod, fake_run_batch)

    # Act
    asyncio.run(main_async([a, b]))

    # Assert
    out = capsys.readouterr().out
    assert "Comparison Summary" in out
    assert "Similarities: 2" in out
    assert "Differences: 1" in out
    assert "Strengths: 1" in out
    assert "Weaknesses: 2" in out
    assert "First difference:" in out and "diff1" in out
