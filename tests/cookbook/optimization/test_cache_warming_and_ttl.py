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
def test_cache_warming_and_ttl_prints_tokens(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    make_env: Callable[..., dict[str, Any]],
) -> None:
    # Arrange files
    (tmp_path / "d1.txt").write_text("x")
    (tmp_path / "d2.txt").write_text("y")

    mod: Any = load_recipe_module("cookbook/optimization/cache-warming-and-ttl.py")
    main_async = mod.main_async

    warm = make_env(status="ok", usage={"total_token_count": 120}, metrics={})
    reuse = make_env(status="ok", usage={"total_token_count": 60}, metrics={})

    count = 0

    async def fake_run(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal count
        out = warm if count == 0 else reuse
        count += 1
        return out

    # Patch the internal _run helper (defined in recipe)
    mod._run = fake_run

    # Act
    asyncio.run(main_async(Path(tmp_path), cache_key="ttl-key"))

    # Assert
    out = capsys.readouterr().out
    assert "Caching enabled:" in out
    assert "RESULTS" in out
    # New output shows provider vs effective totals with status/tokens
    assert "Provider totals (as reported):" in out
    assert "Warm:" in out and "status=ok" in out and "tokens=120" in out
    assert "Reuse:" in out and "status=ok" in out and "tokens=60" in out
    assert "Effective totals (excluding cached content" in out
    assert "Estimated savings:" in out and "%" in out
