from __future__ import annotations

from tests.cookbook.support import load_recipe_module


def test_tok_and_cache_hits_helpers():
    mod = load_recipe_module("cookbook/optimization/context-caching-explicit.py")
    _tok = mod._tok
    _cache_hits = mod._cache_hits

    # _tok tolerant to bad inputs
    assert _tok({}) == 0
    assert _tok({"usage": {}}) == 0
    assert _tok({"usage": {"total_token_count": None}}) == 0

    # _cache_hits counts only truthy dict entries
    env = {
        "metrics": {
            "per_call_meta": [{"cache_applied": True}, {}, {"cache_applied": 1}]
        }
    }
    assert _cache_hits(env) == 2
