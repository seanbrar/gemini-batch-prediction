from __future__ import annotations

from tests.cookbook.support import load_recipe_module


def test_tokens_handles_missing_and_non_int():
    mod = load_recipe_module("cookbook/optimization/cache-warming-and-ttl.py")
    _tokens = mod._tokens

    assert _tokens({}) == 0
    assert _tokens({"total_token_count": None}) == 0
    assert _tokens({"total_token_count": "not-a-number"}) == 0
