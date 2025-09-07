from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_analyze_single_paper_cli_calls_main_async(
    tmp_path: Path,
    cap_cli: Callable[[Any, list[str], Callable[[], Any]], tuple[str, str]],
) -> None:
    mod: Any = load_recipe_module("cookbook/getting-started/analyze-single-paper.py")

    called: dict[str, Any] = {"p": None, "prompt": None}

    async def fake_main_async(path: Path, prompt: str) -> None:
        called["p"] = path
        called["prompt"] = prompt

    mod.main_async = fake_main_async

    f = tmp_path / "doc.txt"
    f.write_text("hello")

    out, _ = cap_cli(mod, ["prog", "--input", str(f), "--prompt", "Do it"], mod.main)

    assert called["p"] == f
    assert called["prompt"] == "Do it"
    assert "Note: Larger files" in out


@pytest.mark.unit
@pytest.mark.cookbook
def test_analyze_single_paper_cli_errors_on_missing_file(
    tmp_path: Path,
    cap_cli: Callable[[Any, list[str], Callable[[], Any]], tuple[str, str]],
) -> None:
    mod: Any = load_recipe_module("cookbook/getting-started/analyze-single-paper.py")
    missing = tmp_path / "nope.txt"
    with pytest.raises(SystemExit):
        cap_cli(mod, ["prog", "--input", str(missing)], mod.main)


@pytest.mark.unit
@pytest.mark.cookbook
def test_cli_uses_demo_dir_when_no_input(
    tmp_path: Path,
    cap_cli: Callable[[Any, list[str], Callable[[], Any]], tuple[str, str]],
) -> None:
    mod: Any = load_recipe_module("cookbook/getting-started/analyze-single-paper.py")

    # Prepare a demo dir with a valid file
    demo = tmp_path / "demo"
    demo.mkdir()
    f = demo / "doc.txt"
    f.write_text("hello")

    # Patch module constants and main_async to capture call
    mod.DEFAULT_TEXT_DEMO_DIR = demo

    called: dict[str, Any] = {"path": None, "prompt": None}

    async def fake_main_async(path: Path, prompt: str) -> None:
        called["path"] = path
        called["prompt"] = prompt

    mod.main_async = fake_main_async

    out, _ = cap_cli(mod, ["prog"], mod.main)  # use default prompt

    assert called["path"] == f
    assert isinstance(called["prompt"], str) and len(called["prompt"]) > 0
    assert "Note: Larger files" in out


@pytest.mark.unit
@pytest.mark.cookbook
def test_cli_errors_when_demo_dir_has_no_matches(
    tmp_path: Path,
    cap_cli: Callable[[Any, list[str], Callable[[], Any]], tuple[str, str]],
) -> None:
    mod: Any = load_recipe_module("cookbook/getting-started/analyze-single-paper.py")

    demo = tmp_path / "demo"
    demo.mkdir()
    # Put a non-matching extension
    (demo / "x.md").write_text("nope")
    mod.DEFAULT_TEXT_DEMO_DIR = demo

    with pytest.raises(SystemExit):
        cap_cli(mod, ["prog"], mod.main)  # no --input
