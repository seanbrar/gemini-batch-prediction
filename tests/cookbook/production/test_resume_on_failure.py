from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.unit
@pytest.mark.cookbook
def test_resume_on_failure_updates_manifest_and_writes_results(
    tmp_path: Path,
    patch_run_batch: Callable[..., None],
    make_env: Callable[..., dict[str, Any]],
) -> None:
    # Arrange: create a few files and load recipe module
    (tmp_path / "one.txt").write_text("one")
    (tmp_path / "two.txt").write_text("two")

    mod: Any = load_recipe_module("cookbook/production/resume-on-failure.py")
    _default_items_from_directory = mod._default_items_from_directory
    run_resume = mod.run_resume

    # Redirect outputs/ paths to a temp sandbox
    manifest_path = tmp_path / "man/manifest.json"
    outputs_dir = tmp_path / "out/items"
    mod.MANIFEST_PATH = manifest_path
    mod.OUTPUTS_DIR = outputs_dir

    # Fake run_batch to return an envelope with an answer and metrics
    async def fake_run_batch(*_args, **_kwargs):
        return make_env(status="ok", answers=["answer"], metrics={"durations": {}})

    patch_run_batch(mod, fake_run_batch)

    # Build items using helper
    items = _default_items_from_directory(tmp_path, "Prompt")

    # Act: run resume
    merged = asyncio.run(run_resume(items=items, failed_only=False, max_retries=0))

    # Assert: manifest written, results written, statuses set to ok
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert len(data) == len(items)
    for it in merged:
        assert it.status == "ok"
        assert it.result_path is not None and Path(it.result_path).exists()
