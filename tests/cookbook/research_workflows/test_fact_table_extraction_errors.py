from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_fact_table_extraction_reports_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mod: Any = load_recipe_module(
        "cookbook/research-workflows/fact-table-extraction.py"
    )
    main_async = mod.main_async

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("one")
    b.write_text("two")
    outdir = tmp_path / "out"

    # Patch extract_one to succeed for 'a' and raise for 'b'
    async def fake_extract_one(path: Path) -> tuple[list[Any], dict[str, Any], None]:
        if path.name == "b.txt":
            raise RuntimeError("boom")
        Finding = mod.Finding
        return (
            [
                Finding(
                    paper_title="t",
                    dataset="d",
                    metric="m",
                    value="v",
                    method="x",
                    notes="n",
                )
            ],
            {},
            None,
        )

    mod.extract_one = fake_extract_one

    asyncio.run(main_async(tmp_path, outdir))
    out = capsys.readouterr().out
    assert "Wrote 1 rows" in out and "files had errors" in out
