from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.cookbook.support import load_recipe_module


@pytest.mark.unit
@pytest.mark.cookbook
def test_observability_prints_no_stage_durations_when_missing(
    tmp_path, capsys, patch_run_batch
):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("y")

    mod: Any = load_recipe_module("cookbook/production/observability-telemetry.py")
    main_async = mod.main_async

    async def fake_run_batch(*_args, **_kwargs):
        return {"status": "ok", "metrics": {}}

    patch_run_batch(mod, fake_run_batch)
    asyncio.run(main_async(Path(tmp_path)))

    out = capsys.readouterr().out
    assert "Status: ok" in out
    assert "No stage durations available" in out
