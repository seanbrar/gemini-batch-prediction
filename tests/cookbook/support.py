from __future__ import annotations

import importlib.util
from pathlib import Path
import runpy
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from types import ModuleType


def load_recipe(rel_path: str) -> Mapping[str, Any]:
    """Load a cookbook recipe module by relative path using runpy.

    Example: load_recipe("cookbook/getting-started/analyze-single-paper.py")
    Returns the module globals dict; access callables like ns["main_async"].
    """
    root = Path(__file__).resolve().parents[2]
    return runpy.run_path(str(root / rel_path))


def load_recipe_module(rel_path: str) -> ModuleType:
    """Load a cookbook recipe as a real module to allow monkeypatching attributes.

    Returns the imported module object.
    """
    root = Path(__file__).resolve().parents[2]
    file_path = root / rel_path
    mod_name = "cookbook_test_" + rel_path.replace("/", "_").replace("-", "_").replace(
        ".py", ""
    )
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def assert_contains_all(text: str, parts: Iterable[str]) -> None:
    """Assert that all substrings in parts are present in text."""
    missing = [p for p in parts if p not in text]
    assert not missing, f"missing expected text: {missing}"
