"""Source helpers: explicit directory expansion and file walking.

These helpers restore feature completeness for directory inputs while keeping
the core pipeline explicit and type-safe. End-users can call these utilities to
materialize `Source` objects prior to invoking the pipeline/frontdoor.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from .types import Source

if TYPE_CHECKING:
    from collections.abc import Iterable


_EXCLUDE_DIRS: Final[set[str]] = {
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    ".env",
    "build",
    "dist",
    ".tox",
    ".coverage",
}


def iter_files(directory: str | Path) -> Iterable[Path]:
    """Yield all files under `directory` recursively with stable ordering.

    Excludes common VCS/virtualenv/build directories for predictable behavior.

    Raises:
        ValueError: If `directory` does not exist or is not a directory.
    """
    import os

    root_path = Path(directory)
    if not root_path.is_dir():
        raise ValueError("directory must be an existing directory")

    def _walk() -> Iterable[Path]:
        for root, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [d for d in sorted(dirnames) if d not in _EXCLUDE_DIRS]
            for fname in sorted(filenames):
                p = Path(root) / fname
                try:
                    if p.is_file():
                        yield p
                except OSError:
                    continue

    return _walk()


def sources_from_directory(directory: str | Path) -> tuple[Source, ...]:
    """Return `Source` objects for all files under `directory` (recursive).

    Uses `Source.from_file` for MIME detection and lazy loading.

    Raises:
        ValueError: If `directory` does not exist or is not a directory.
    """
    files = list(iter_files(directory))
    return tuple(Source.from_file(p) for p in files)
