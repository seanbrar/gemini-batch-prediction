from __future__ import annotations

import pytest

from cookbook.utils.demo_inputs import (
    resolve_dir_or_exit,
    resolve_file_or_exit,
)


@pytest.mark.unit
@pytest.mark.cookbook
def test_resolve_dir_or_exit_prefers_user_path(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    got = resolve_dir_or_exit(d, fallback=tmp_path / "fallback", hint="H")
    assert got == d


@pytest.mark.unit
@pytest.mark.cookbook
def test_resolve_dir_or_exit_uses_fallback_when_exists(tmp_path):
    fb = tmp_path / "fallback"
    fb.mkdir()
    got = resolve_dir_or_exit(None, fallback=fb, hint="H")
    assert got == fb


@pytest.mark.unit
@pytest.mark.cookbook
def test_resolve_dir_or_exit_errors_when_missing(tmp_path, capsys):
    with pytest.raises(SystemExit) as ei:
        resolve_dir_or_exit(None, fallback=tmp_path / "missing", hint="Use --input")
    assert ei.value.code == 2
    assert "Use --input" in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.cookbook
def test_resolve_file_or_exit_user_path(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    got = resolve_file_or_exit(f, search_dir=tmp_path, exts=[".txt"], hint="H")
    assert got == f


@pytest.mark.unit
@pytest.mark.cookbook
def test_resolve_file_or_exit_pick_from_dir(tmp_path):
    (tmp_path / "b.txt").write_text("b")
    got = resolve_file_or_exit(None, search_dir=tmp_path, exts=["txt"], hint="H")
    assert got.name == "b.txt"


@pytest.mark.unit
@pytest.mark.cookbook
def test_resolve_file_or_exit_errors_when_no_dir(tmp_path, capsys):
    with pytest.raises(SystemExit) as ei:
        resolve_file_or_exit(
            None, search_dir=tmp_path / "missing", exts=[".txt"], hint="No dir"
        )
    assert ei.value.code == 2
    assert "No dir" in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.cookbook
def test_resolve_file_or_exit_errors_when_no_match(tmp_path, capsys):
    (tmp_path / "a.md").write_text("x")
    with pytest.raises(SystemExit) as ei:
        resolve_file_or_exit(None, search_dir=tmp_path, exts=[".txt"], hint="No file")
    assert ei.value.code == 2
    assert "No file" in capsys.readouterr().err


@pytest.mark.unit
def test_pick_files_by_ext_limit_non_positive(tmp_path):
    # Ensure function guards to at least 1 when limit <= 0
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    from cookbook.utils.demo_inputs import pick_files_by_ext

    got0 = pick_files_by_ext(tmp_path, [".txt"], limit=0)
    got_neg = pick_files_by_ext(tmp_path, [".txt"], limit=-3)
    assert len(got0) == 1 and len(got_neg) == 1
