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
def test_observability_prints_status_durations_and_hints(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    patch_run_batch: Callable[..., None],
    make_env: Callable[..., dict[str, Any]],
) -> None:
    # Arrange test files
    (tmp_path / "f1.txt").write_text("a")
    (tmp_path / "f2.txt").write_text("b")

    mod: Any = load_recipe_module("cookbook/production/observability-telemetry.py")
    main_async = mod.main_async

    fake_env = make_env(
        status="ok",
        metrics={
            "durations": {"stage.prepare": 0.0123, "stage.execute": 0.0456},
            "vectorized_n_calls": 2,
            "parallel_n_calls": 0,
        },
    )

    async def fake_run_batch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return fake_env

    # Patch run_batch in the module namespace dict
    patch_run_batch(mod, fake_run_batch)

    # Act
    asyncio.run(main_async(Path(tmp_path)))

    # Assert
    out = capsys.readouterr().out
    assert "Status: ok" in out
    assert "Stage durations" in out and "stage.prepare" in out
    assert "Vectorized API calls" in out
    assert "Tip: For exports" in out
