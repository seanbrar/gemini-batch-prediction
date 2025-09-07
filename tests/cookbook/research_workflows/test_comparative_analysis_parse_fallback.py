from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.cookbook.support import load_recipe_module


@pytest.mark.unit
@pytest.mark.cookbook
def test_comparative_analysis_fallback_prints_raw_when_not_json(tmp_path, capsys):
    mod: Any = load_recipe_module("cookbook/research-workflows/comparative-analysis.py")

    async def fake_run_batch(*_args, **_kwargs):
        return {"answers": ["not-json"]}

    mod.run_batch = fake_run_batch

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("x")
    b.write_text("y")
    asyncio.run(mod.main_async([a, b]))
    out = capsys.readouterr().out
    assert "Could not parse JSON" in out or "⚠️ Could not parse JSON" in out
