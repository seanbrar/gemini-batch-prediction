from __future__ import annotations

import pytest

from cookbook.utils.demo_inputs import pick_file_by_ext, pick_files_by_ext


@pytest.mark.unit
def test_pick_file_by_ext_returns_first_match(tmp_path):
    (tmp_path / "a.md").write_text("nope")
    (tmp_path / "b.txt").write_text("ok1")
    (tmp_path / "c.txt").write_text("ok2")
    got = pick_file_by_ext(tmp_path, [".txt"])
    assert got is not None
    # Sorted order means b.txt comes before c.txt
    assert got.name == "b.txt"


@pytest.mark.unit
def test_pick_files_by_ext_respects_limit_and_order(tmp_path):
    (tmp_path / "z.txt").write_text("z")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "m.txt").write_text("m")
    got = pick_files_by_ext(tmp_path, ["txt"], limit=2)
    assert [p.name for p in got] == ["a.txt", "m.txt"]
