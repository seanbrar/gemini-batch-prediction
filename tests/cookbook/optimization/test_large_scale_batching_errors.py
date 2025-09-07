from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.cookbook.support import load_recipe_module


@pytest.mark.unit
@pytest.mark.cookbook
def test_large_scale_batching_prints_errors_when_present(
    tmp_path, capsys, patch_run_parallel
):
    (tmp_path / "a.txt").write_text("a")

    mod: Any = load_recipe_module("cookbook/optimization/large-scale-batching.py")

    async def fake_run_parallel(_prompt, sources, **_kwargs):
        return {
            "status": "ok",
            "answers": ["x" for _ in sources],
            "metrics": {"parallel_n_calls": len(tuple(sources)), "parallel_errors": 2},
        }

    patch_run_parallel(mod, fake_run_parallel)

    asyncio.run(mod.main_async(Path(tmp_path), prompt="Q", concurrency=2))
    out = capsys.readouterr().out
    assert "Errors:" in out and "2" in out
