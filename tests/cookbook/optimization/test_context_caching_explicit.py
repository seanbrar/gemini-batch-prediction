from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.cookbook.support import load_recipe_module


@pytest.mark.unit
@pytest.mark.cookbook
def test_context_caching_prints_savings_and_hits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange: create two files so the recipe finds sources
    (tmp_path / "doc1.txt").write_text("hello")
    (tmp_path / "doc2.txt").write_text("world")

    # Load recipe and patch its internal _run function to yield warm then reuse payloads
    mod: Any = load_recipe_module("cookbook/optimization/context-caching-explicit.py")
    main_async = mod.main_async

    warm_env = {"usage": {"total_token_count": 100}, "metrics": {"per_call_meta": []}}
    reuse_env = {
        "usage": {"total_token_count": 40},
        "metrics": {
            "per_call_meta": [{"cache_applied": True}, {"cache_applied": False}]
        },
    }

    calls: list[dict[str, Any]] = []

    async def fake_run(prompts: Any, sources: Any, **_kwargs: Any) -> dict[str, Any]:
        # record the call, return warm first then reuse
        calls.append({"prompts": list(prompts), "n_sources": len(tuple(sources))})
        return warm_env if len(calls) == 1 else reuse_env

    mod._run = fake_run

    # Act
    asyncio.run(main_async(Path(tmp_path), cache_key="k1"))

    # Assert: savings and cache hits are reported; two calls were made
    out = capsys.readouterr().out
    assert "Warming cache" in out and "Reusing cache" in out
    # New output format emphasizes provider vs effective totals
    assert "Provider totals (as reported):" in out
    assert "Warm total tokens:" in out and "Reuse total tokens:" in out
    assert "Reported savings:" in out and "%" in out
    assert "Effective totals (excluding cached content" in out
    assert "Estimated savings:" in out and "%" in out
    assert "Cache applied on:" in out
    assert len(calls) == 2
