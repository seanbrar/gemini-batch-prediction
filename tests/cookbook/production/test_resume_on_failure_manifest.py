from __future__ import annotations

import pytest

from tests.cookbook.support import load_recipe_module


@pytest.mark.unit
@pytest.mark.cookbook
def test_manifest_save_and_load_roundtrip(tmp_path):
    mod = load_recipe_module("cookbook/production/resume-on-failure.py")
    Item = mod.Item
    _load_manifest = mod._load_manifest
    _save_manifest = mod._save_manifest

    path = tmp_path / "manifest.json"
    # Empty when missing
    assert _load_manifest(path) == []

    items = [Item(id="a.txt", source_path="/p/a.txt", prompt="P", status="ok")]
    _save_manifest(path, items)
    loaded = _load_manifest(path)
    assert [i.id for i in loaded] == ["a.txt"]
