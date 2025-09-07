from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_multi_format_pipeline_prints_status_and_synthesis(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    patch_run_batch: Callable[..., None],
    make_env: Callable[..., dict[str, Any]],
) -> None:
    # Arrange: create three different files
    pdf = tmp_path / "doc.pdf"
    img = tmp_path / "img.png"
    vid = tmp_path / "clip.mp4"
    for p in (pdf, img, vid):
        p.write_bytes(b"\x00")

    mod: Any = load_recipe_module("cookbook/optimization/multi-format-pipeline.py")
    main_async = mod.main_async

    async def fake_run_batch(*_args, **_kwargs):
        return make_env(
            status="ok", answers=["ans1", "ans2", "final synthesis text..."]
        )

    patch_run_batch(mod, fake_run_batch)

    # Act
    asyncio.run(main_async(pdf, img, vid, base_dir=None))

    # Assert
    out = capsys.readouterr().out
    assert "Status: ok" in out
    assert "Synthesis (first 500 chars)" in out
