from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from tests.cookbook.support import load_recipe_module


@pytest.mark.unit
@pytest.mark.cookbook
def test_resume_on_failure_tracks_errors_and_retries(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "f.txt").write_text("x")
    mod: Any = load_recipe_module("cookbook/production/resume-on-failure.py")
    run_resume = mod.run_resume
    _default_items_from_directory = mod._default_items_from_directory

    # Redirect outputs under temp dir
    mod.MANIFEST_PATH = tmp_path / "man/manifest.json"
    mod.OUTPUTS_DIR = tmp_path / "out/items"

    async def fake_run_batch(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("boom")

    mod.run_batch = fake_run_batch

    items = _default_items_from_directory(tmp_path, "Prompt")
    merged = asyncio.run(
        run_resume(items=items, failed_only=False, max_retries=1, backoff_s=0.0)
    )

    assert (tmp_path / "man/manifest.json").exists()
    assert any(i.status == "error" for i in merged)
    assert any((i.retries >= 1 and i.error) for i in merged)
