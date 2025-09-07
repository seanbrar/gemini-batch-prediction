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
def test_rate_limits_and_concurrency_prints_both_summaries(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    patch_run_batch: Callable[..., None],
    make_env: Callable[..., dict[str, Any]],
) -> None:
    # Arrange files so the recipe finds sources
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")

    mod: Any = load_recipe_module("cookbook/production/rate-limits-and-concurrency.py")
    main_async = mod.main_async

    calls = {"n": 0}

    async def fake_run_batch(*_args, **_kwargs):
        calls["n"] += 1
        # First call -> sequential; second call -> bounded fan-out
        if calls["n"] == 1:
            return make_env(
                answers=["a", "b", "c"],
                metrics={
                    "concurrency_used": 1,
                    "per_call_meta": [{"duration_s": 0.1}],
                },
            )
        return make_env(
            answers=["a", "b", "c"],
            metrics={"concurrency_used": 4, "per_call_meta": [{"duration_s": 0.05}]},
        )

    patch_run_batch(mod, fake_run_batch)

    # Act
    asyncio.run(main_async(Path(tmp_path), concurrency=4))

    # Assert
    out = capsys.readouterr().out
    assert "Sequential (concurrency=1)" in out
    assert "Answers:" in out and "concurrency_used:" in out
    assert "Bounded fan-out (concurrency=4)" in out
