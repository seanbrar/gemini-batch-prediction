"""Internal helpers for development-time feature flags.

This module intentionally stays minimal to preserve simplicity and avoid
cross-module coupling. It centralizes how we check opt-in validation toggles
so semantics remain consistent across the codebase.
"""

from __future__ import annotations

import os

__all__ = ["dev_validate_enabled"]


def dev_validate_enabled(*, override: bool | None = None) -> bool:
    """Return True when dev-time validation is enabled.

    - If ``override`` is provided, it takes precedence.
    - Otherwise, returns True when the environment variable
      ``GEMINI_PIPELINE_VALIDATE`` is exactly ``"1"``.
    """
    if override is not None:
        return bool(override)
    return os.getenv("GEMINI_PIPELINE_VALIDATE") == "1"
