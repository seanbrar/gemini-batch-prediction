from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.cookbook.support import load_recipe_module


@pytest.mark.unit
@pytest.mark.cookbook
def test_large_scale_batching_prints_status_and_parallel_calls(
    tmp_path, capsys, patch_run_parallel
):
    # Arrange
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")

    mod: Any = load_recipe_module("cookbook/optimization/large-scale-batching.py")
    main_async = mod.main_async

    async def fake_run_parallel(_prompt, sources, **_kwargs):
        return {
            "status": "ok",
            "answers": ["x" for _ in sources],
            "metrics": {"parallel_n_calls": len(tuple(sources)), "parallel_errors": 0},
        }

    patch_run_parallel(mod, fake_run_parallel)

    # Act
    asyncio.run(main_async(Path(tmp_path), prompt="Q", concurrency=3))

    # Assert
    out = capsys.readouterr().out
    assert "Status: ok" in out
    assert "Parallel calls:" in out
