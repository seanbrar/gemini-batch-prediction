from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_comparative_analysis_cli_errors_when_insufficient_files(
    tmp_path: Path,
    cap_cli: Callable[[Any, list[str], Callable[[], Any]], tuple[str, str]],
) -> None:
    # Only one file present; CLI should error when trying to auto-pick 2
    (tmp_path / "a.txt").write_text("x")

    mod = load_recipe_module("cookbook/research-workflows/comparative-analysis.py")
    with pytest.raises(SystemExit):
        cap_cli(mod, ["prog", "--input", str(tmp_path)], mod.main)
