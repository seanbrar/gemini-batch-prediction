from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tests.cookbook.support import load_recipe_module


@pytest.mark.unit
@pytest.mark.cookbook
def test_iter_files_filters_media_types(tmp_path):
    mod: Any = load_recipe_module(
        "cookbook/research-workflows/fact-table-extraction.py"
    )
    _iter_files = mod._iter_files

    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "img.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "clip.mp4").write_bytes(b"\x00\x00")

    got = list(_iter_files(tmp_path))
    names = {p.name for p in got}
    assert names == {"a.txt", "b.pdf"}


@pytest.mark.unit
@pytest.mark.cookbook
def test_try_parse_findings_handles_list_and_dict():
    mod: Any = load_recipe_module(
        "cookbook/research-workflows/fact-table-extraction.py"
    )
    Finding = mod.Finding
    _try_parse_findings = mod._try_parse_findings

    one = {
        "paper_title": "t",
        "dataset": "d",
        "metric": "m",
        "value": "v",
        "method": "x",
        "notes": "n",
    }
    as_dict = json.dumps(one)
    as_list = json.dumps([one, one])

    got1 = _try_parse_findings(as_dict)
    got2 = _try_parse_findings(as_list)

    assert [isinstance(x, Finding) for x in got1] == [True]
    assert [isinstance(x, Finding) for x in got2] == [True, True]


@pytest.mark.unit
@pytest.mark.cookbook
def test_extract_one_fallback_when_not_json(tmp_path):
    mod: Any = load_recipe_module(
        "cookbook/research-workflows/fact-table-extraction.py"
    )
    extract_one = mod.extract_one

    # Patch run_batch to yield a non-JSON answer
    async def fake_run_batch(*_args, **_kwargs):
        return {"answers": ["not-json"], "metrics": {}}

    mod.run_batch = fake_run_batch

    f = tmp_path / "p.txt"
    f.write_text("x")
    rows, metrics, err = asyncio.run(extract_one(f))
    assert rows and rows[0].notes == "fallback_parse"
    assert metrics == {}
    assert err is None
