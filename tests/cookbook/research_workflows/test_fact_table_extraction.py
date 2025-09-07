from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_fact_table_extraction_writes_jsonl_and_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange two files
    (tmp_path / "p1.txt").write_text("one")
    (tmp_path / "p2.txt").write_text("two")
    outdir = tmp_path / "out"

    mod: Any = load_recipe_module(
        "cookbook/research-workflows/fact-table-extraction.py"
    )
    main_async = mod.main_async
    Finding = mod.Finding

    async def fake_extract_one(path: Path) -> tuple[list[Any], dict[str, Any], None]:
        return (
            [
                Finding(
                    paper_title=path.stem,
                    dataset="ds",
                    metric="acc",
                    value="0.9",
                    method="m",
                    notes="n",
                )
            ],
            {},
            None,
        )

    mod.extract_one = fake_extract_one

    # Act
    asyncio.run(main_async(tmp_path, outdir))

    # Assert
    jsonl = outdir / "facts.jsonl"
    csv = outdir / "facts.csv"
    assert jsonl.exists() and csv.exists()
    # Two rows expected (one per input file)
    assert sum(1 for _ in jsonl.open()) == 2
    text = jsonl.read_text()
    assert "paper_title" in text and "dataset" in text
    out = capsys.readouterr().out
    assert "Wrote 2 rows" in out
