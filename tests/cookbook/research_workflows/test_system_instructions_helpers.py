from __future__ import annotations

from tests.cookbook.support import load_recipe_module


def test_system_instructions_helpers_cover_builder_and_cfg():
    mod = load_recipe_module(
        "cookbook/research-workflows/system-instructions-with-research-helper.py"
    )
    builder = mod.default_aggregate_prompt_builder
    text = builder(["Q1", "Q2"])  # expect two numbered items
    assert "exactly 2 items" in text and "1. Q1" in text and "2. Q2" in text

    mk = mod.make_cfg_with_system
    cfg = mk(system="Be precise.")
    # Overrides land in the extra mapping for FrozenConfig
    assert cfg.extra.get("prompts.system") == "Be precise."
    assert cfg.extra.get("prompts.sources_policy") in ("append_or_replace",)
