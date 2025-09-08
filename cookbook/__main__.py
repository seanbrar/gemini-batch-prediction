"""Cookbook runner module.

Run cookbook recipes without setting PYTHONPATH, supporting either
path-like specs (relative to the cookbook folder) or a dotted-like
spec that maps underscores to hyphens.

Examples:
- python -m cookbook getting-started/analyze-single-paper.py -- --limit 1
- python -m cookbook production.resume_on_failure -- --limit 2

Use "--" to pass arguments through to the recipe script.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import runpy
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


COOKBOOK_DIRNAME = "cookbook"
EXCLUDE_DIRS = {"utils", "templates", "data", "__pycache__"}


@dataclass(frozen=True)
class RecipeSpec:
    """A resolved recipe specification.

    Attributes:
        path: Absolute path to the recipe Python file.
        display: Human-readable identifier shown to the user.
    """

    path: Path
    display: str


def repo_root() -> Path:
    """Return the repository root directory.

    Assumes this file lives under <root>/cookbook/__main__.py.
    """

    return Path(__file__).resolve().parents[1]


def cookbook_root() -> Path:
    """Return the absolute path to the cookbook directory."""

    return repo_root() / COOKBOOK_DIRNAME


def is_recipe_file(path: Path) -> bool:
    """True if the path looks like a runnable recipe file."""

    name = path.name
    return name.endswith(".py") and name not in {"__init__.py", "__main__.py"}


def list_recipes() -> list[RecipeSpec]:
    """Discover recipe files under the cookbook directory.

    Excludes helper directories like utils/, templates/, and data/.
    """

    root = cookbook_root()
    results: list[RecipeSpec] = []
    for path in root.rglob("*.py"):
        # Skip non-recipe directories
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if not is_recipe_file(path):
            continue
        rel = path.relative_to(root)
        results.append(RecipeSpec(path=path, display=str(rel)))
    results.sort(key=lambda s: s.display)
    return results


def dotted_to_path(spec: str) -> str:
    """Convert a dotted-like spec to a cookbook-relative path.

    Behavior:
    - Replace '.' with '/'
    - Replace '_' with '-' (to match on-disk naming)
    - Append '.py' if missing
    """

    candidate = spec.replace(".", "/").replace("_", "-")
    if not candidate.endswith(".py"):
        candidate += ".py"
    return candidate


def resolve_spec(spec: str) -> RecipeSpec:
    """Resolve a user-provided spec into an absolute file path.

    Accepts:
    - A path relative to repo root (e.g., 'cookbook/production/foo.py')
    - A path relative to cookbook (e.g., 'production/foo.py')
    - A dotted-like spec (e.g., 'production.foo_bar' → 'production/foo-bar.py')
    """

    root = repo_root()
    croot = cookbook_root()

    # Helper to validate a candidate path strictly belongs to cookbook and is a recipe
    def _accept(path: Path) -> RecipeSpec | None:
        if not path.is_file():
            return None
        if path.suffix != ".py":
            return None
        if not path.is_relative_to(croot):
            return None
        if not is_recipe_file(path):
            return None
        rel = path.relative_to(croot)
        return RecipeSpec(path=path, display=str(rel))

    # First try as-is relative to repo root, but only accept cookbook recipes
    p = (root / spec).resolve()
    accepted = _accept(p)
    if accepted is not None:
        return accepted

    # If it starts with 'cookbook/' (or Windows 'cookbook\'), strip and try relative to cookbook
    if spec.startswith((COOKBOOK_DIRNAME + "/", COOKBOOK_DIRNAME + "\\")):
        rel = spec[len(COOKBOOK_DIRNAME) + 1 :]
        p2 = (croot / rel).resolve()
        accepted2 = _accept(p2)
        if accepted2 is not None:
            return accepted2

    # Try dotted-like conversion
    mapped = dotted_to_path(spec)
    p3 = (croot / mapped).resolve()
    accepted3 = _accept(p3)
    if accepted3 is not None:
        return accepted3

    # Try path-like relative to cookbook (without '.py')
    alt = spec
    if not alt.endswith(".py"):
        alt += ".py"
    p4 = (croot / alt).resolve()
    accepted4 = _accept(p4)
    if accepted4 is not None:
        return accepted4

    raise FileNotFoundError(
        "Recipe not found. Tried: "
        f"{p}, {croot / spec}, {p3}, {p4}. "
        "Provide a path relative to the repo or cookbook, or a dotted-like spec. "
        "Use --list to view available recipes."
    )


def print_recipe_list(recipes: Iterable[RecipeSpec]) -> None:
    """Print available recipes in a compact, user-friendly format."""

    print("Available recipes (relative to 'cookbook/'):")
    for r in recipes:
        dotted = r.display.replace("/", ".").replace("-", "_").removesuffix(".py")
        print(f"  - {r.display}    (dotted: {dotted})")


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse runner arguments.

    Returns (namespace, passthrough_args). Passthrough args are the
    arguments intended for the recipe, captured after a "--".
    """

    parser = argparse.ArgumentParser(
        prog="python -m cookbook",
        description=(
            "Run cookbook recipes without setting PYTHONPATH. "
            "Pass recipe args after '--'."
        ),
    )
    parser.add_argument(
        "spec",
        nargs="?",
        help=(
            "Recipe spec. Examples: "
            "'getting-started/analyze-single-paper.py' or "
            "'production.resume_on_failure'"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available recipes and exit",
    )
    # Default to running from the repo root for least astonishment.
    # Users can opt out with --no-cwd-repo-root.
    parser.add_argument(
        "--cwd-repo-root",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Change to the repository root before executing the recipe"
            " (default: enabled)"
        ),
    )

    if "--" in argv:
        idx = argv.index("--")
        runner_args = list(argv[:idx])
        passthrough = list(argv[idx + 1 :])
    else:
        runner_args = list(argv)
        passthrough = []

    ns = parser.parse_args(runner_args)
    return ns, passthrough


def run_recipe(
    recipe: RecipeSpec, passthrough: Sequence[str], *, cwd_repo_root: bool = False
) -> int:
    """Execute the recipe in-process using runpy.

    Sets sys.argv to mimic direct script execution.
    Returns exit status code (0 for success).
    """

    # Prepare argv for the recipe
    prev_argv = list(sys.argv)
    sys.argv = [str(recipe.path), *list(passthrough)]
    # Optionally run from the repo root for least astonishment in relative paths
    old_cwd = Path.cwd()
    try:
        if cwd_repo_root:
            os.chdir(str(repo_root()))
        runpy.run_path(str(recipe.path), run_name="__main__")
    finally:
        if cwd_repo_root:
            os.chdir(str(old_cwd))
        # Restore sys.argv to avoid leaking state
        sys.argv = prev_argv
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args, passthrough = parse_args(list(argv) if argv is not None else sys.argv[1:])

    if args.list:
        print_recipe_list(list_recipes())
        return 0

    if not args.spec:
        # Show available recipes to reduce friction instead of erroring out
        print_recipe_list(list_recipes())
        return 0

    try:
        spec = resolve_spec(args.spec)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    return run_recipe(spec, passthrough, cwd_repo_root=args.cwd_repo_root)


if __name__ == "__main__":  # pragma: no cover - direct execution guard
    raise SystemExit(main())
