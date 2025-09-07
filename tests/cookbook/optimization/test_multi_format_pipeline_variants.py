from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.cookbook.support import load_recipe_module


@pytest.mark.unit
@pytest.mark.cookbook
def test_multi_format_pipeline_picks_from_base_dir(tmp_path, capsys):
    mod: Any = load_recipe_module("cookbook/optimization/multi-format-pipeline.py")

    base = tmp_path / "media"
    base.mkdir()
    pdf = base / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    async def fake_run_batch(*_args, **_kwargs):
        return {"status": "ok", "answers": ["a", "b", "c"]}

    mod.run_batch = fake_run_batch

    asyncio.run(mod.main_async(None, None, None, base))
    out = capsys.readouterr().out
    assert "Status: ok" in out and "Synthesis (first 500 chars)" in out


@pytest.mark.unit
@pytest.mark.cookbook
def test_multi_format_pipeline_errors_when_no_existing_inputs(tmp_path):
    mod: Any = load_recipe_module("cookbook/optimization/multi-format-pipeline.py")
    # Pass non-existent paths for all three so base_dir fallback is not used
    with pytest.raises(SystemExit):
        asyncio.run(
            mod.main_async(
                tmp_path / "missing.pdf", tmp_path / "no.png", tmp_path / "no.mp4", None
            )
        )
