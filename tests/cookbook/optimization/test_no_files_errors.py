from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.cookbook.support import load_recipe_module

Case = tuple[str, Callable[[Any, Path], Any]]


def _call_large_scale(mod: Any, d: Path) -> Any:
    return mod.main_async(d, prompt="Q", concurrency=2)


def _call_context_cache(mod: Any, d: Path) -> Any:
    return mod.main_async(d, cache_key="k")


def _call_cache_ttl(mod: Any, d: Path) -> Any:
    return mod.main_async(d, cache_key="k")


@pytest.mark.unit
@pytest.mark.cookbook
@pytest.mark.parametrize(
    "recipe,invoker",
    [
        ("cookbook/optimization/large-scale-batching.py", _call_large_scale),
        ("cookbook/optimization/context-caching-explicit.py", _call_context_cache),
        ("cookbook/optimization/cache-warming-and-ttl.py", _call_cache_ttl),
    ],
)
def test_recipes_error_when_no_files(
    tmp_path: Path, recipe: str, invoker: Callable[[Any, Path], Any]
) -> None:
    mod: Any = load_recipe_module(recipe)
    with pytest.raises(SystemExit):
        asyncio.run(invoker(mod, tmp_path))
