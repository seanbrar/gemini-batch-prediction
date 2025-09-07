from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.cookbook.support import load_recipe_module


@pytest.mark.unit
@pytest.mark.cookbook
def test_process_item_writes_partial_and_result_file(tmp_path):
    mod: Any = load_recipe_module("cookbook/production/resume-on-failure.py")
    Item = mod.Item
    _process_item = mod._process_item

    # Redirect outputs
    mod.OUTPUTS_DIR = tmp_path / "out"

    item = Item(id="a.txt", source_path=str(tmp_path / "a.txt"), prompt="P")
    Path(item.source_path).write_text("x")

    async def fake_run_batch(*_args, **_kwargs):
        return {"status": "partial", "answers": ["ans"], "metrics": {}}

    mod.run_batch = fake_run_batch

    updated = asyncio.run(_process_item(item))
    assert updated.status == "partial"
    assert updated.result_path is not None and Path(updated.result_path).exists()
