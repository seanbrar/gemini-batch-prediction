from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.cookbook.support import load_recipe_module


@pytest.mark.unit
@pytest.mark.cookbook
def test_literature_synthesis_parse_fallback_prints_raw(tmp_path, capsys):
    # Arrange files
    (tmp_path / "p1.txt").write_text("one")
    (tmp_path / "p2.txt").write_text("two")
    mod: Any = load_recipe_module("cookbook/research-workflows/literature-synthesis.py")
    main_async = mod.main_async

    async def fake_run_batch(*_args, **_kwargs):
        return {"answers": ["not-json"]}

    mod.run_batch = fake_run_batch
    asyncio.run(main_async(Path(tmp_path)))

    out = capsys.readouterr().out
    assert "Structured parse failed" in out
    assert "first 400 chars" in out
