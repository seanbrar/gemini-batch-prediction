from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.unit
@pytest.mark.cookbook
def test_multi_format_pipeline_cli_calls_main_async(
    tmp_path: Path,
    cap_cli: Callable[[Any, list[str], Callable[[], Any]], tuple[str, str]],
) -> None:
    mod: Any = load_recipe_module("cookbook/optimization/multi-format-pipeline.py")
    called = {"pdf": None, "img": None, "vid": None, "base": None}

    async def fake_main_async(pdf, image, video, base_dir):
        called["pdf"], called["img"], called["vid"], called["base"] = (
            pdf,
            image,
            video,
            base_dir,
        )

    mod.main_async = fake_main_async

    pdf = tmp_path / "doc.pdf"
    img = tmp_path / "img.png"
    vid = tmp_path / "clip.mp4"
    for p in (pdf, img, vid):
        p.write_bytes(b"\x00")

    argv = ["prog", "--pdf", str(pdf), "--image", str(img), "--video", str(vid)]
    out, _ = cap_cli(mod, argv, mod.main)

    assert called["pdf"] == pdf and called["img"] == img and called["vid"] == vid
    assert "Note: File size/count affect runtime and tokens." in out
