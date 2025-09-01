from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

PUBLIC_DIR = Path("cookbook/data/public")
EXAMPLES_DIR = Path("cookbook/data/research_papers")


def resolve_data_dir(preferred: Path | None = None) -> Path:
    """Return a usable data directory.

    Order: preferred (if exists) → cookbook/data/public → examples/test_data.
    Raises if none found.
    """
    candidates = [c for c in [preferred, PUBLIC_DIR, EXAMPLES_DIR] if c]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No data directory found. Tried: {', '.join(str(c) for c in candidates)}. "
        "Run `make fetch-cookbook-data` or provide a directory explicitly."
    )


def pick_file_by_ext(root: Path, exts: Iterable[str]) -> Path | None:
    """Return the first file under `root` with an allowed extension.

    Extensions are case-insensitive and may include the dot (e.g., '.mp4').
    """
    allowed = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in allowed:
            return p
    return None
