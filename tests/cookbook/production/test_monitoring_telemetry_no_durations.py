from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.unit
@pytest.mark.cookbook
def test_monitoring_telemetry_prints_no_stage_durations_when_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    patch_run_batch: Callable[..., None],
    make_env: Callable[..., dict[str, Any]],
) -> None:
    mod: Any = load_recipe_module("cookbook/production/monitoring-telemetry.py")

    async def fake_run_batch(*_args, **_kwargs):
        return make_env(status="ok", metrics={})

    patch_run_batch(mod, fake_run_batch)
    (tmp_path / "a.txt").write_text("x")
    asyncio.run(mod.main_async(Path(tmp_path)))

    out = capsys.readouterr().out
    assert "Status: ok" in out and "No stage durations available." in out
