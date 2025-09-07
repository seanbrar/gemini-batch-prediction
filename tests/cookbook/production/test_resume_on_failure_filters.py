from __future__ import annotations

import pytest

from tests.cookbook.support import load_recipe_module


@pytest.mark.unit
@pytest.mark.cookbook
def test_is_supported_file_skips_audio_and_video(tmp_path):
    mod = load_recipe_module("cookbook/production/resume-on-failure.py")
    _is_supported_file = mod._is_supported_file

    # Create dummy files with extensions; existence not required by function
    assert _is_supported_file(tmp_path / "doc.txt") is True
    assert _is_supported_file(tmp_path / "paper.pdf") is True
    assert _is_supported_file(tmp_path / "song.mp3") is False
    assert _is_supported_file(tmp_path / "track.wav") is False
    assert _is_supported_file(tmp_path / "clip.mp4") is False


@pytest.mark.unit
@pytest.mark.cookbook
def test_default_items_from_directory_filters_and_ids(tmp_path):
    mod = load_recipe_module("cookbook/production/resume-on-failure.py")
    _default_items_from_directory = mod._default_items_from_directory

    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "song.mp3").write_bytes(b"\x00\x00")  # should be filtered

    items = _default_items_from_directory(tmp_path, "P")
    ids = {i.id for i in items}
    assert "a.txt" in ids and "b.pdf" in ids
    assert all(not item_id.endswith(".mp3") for item_id in ids)
