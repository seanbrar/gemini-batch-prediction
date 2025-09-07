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
def test_monitoring_telemetry_prints_stage_durations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    patch_run_batch: Callable[..., None],
    make_env: Callable[..., dict[str, Any]],
) -> None:
    # Arrange
    (tmp_path / "t1.txt").write_text("one")
    (tmp_path / "t2.txt").write_text("two")

    mod: Any = load_recipe_module("cookbook/production/monitoring-telemetry.py")
    main_async = mod.main_async

    fake_env = make_env(
        status="ok",
        metrics={
            "durations": {"stage.prepare": 0.01, "stage.execute": 0.02},
            "vectorized_n_calls": 1,
        },
    )

    async def fake_run_batch(*_args, **_kwargs):
        return fake_env

    patch_run_batch(mod, fake_run_batch)

    # Act
    asyncio.run(main_async(Path(tmp_path)))

    # Assert
    out = capsys.readouterr().out
    assert "Status: ok" in out
    assert "Stage durations" in out
    assert "stage.prepare" in out and "stage.execute" in out


@pytest.mark.unit
@pytest.mark.cookbook
def test_monitoring_telemetry_errors_when_no_files(tmp_path: Path) -> None:
    mod: Any = load_recipe_module("cookbook/production/monitoring-telemetry.py")
    with pytest.raises(SystemExit):
        asyncio.run(mod.main_async(Path(tmp_path)))
