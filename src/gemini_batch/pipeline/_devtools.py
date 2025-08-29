"""Developer utilities for advanced pipeline composition (internal).

This module exposes helpers that are useful when authoring extensions or
composing custom pipelines. It is not part of the stable public API and may
change between minor versions.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gemini_batch.core.exceptions import GeminiBatchError

    from .base import BaseAsyncHandler

log = logging.getLogger(__name__)


def compose_pipeline(
    *handlers: BaseAsyncHandler[Any, Any, GeminiBatchError],
) -> list[BaseAsyncHandler[Any, Any, GeminiBatchError]]:
    """Validate and return a typed handler sequence for executor use.

    - Confirms each handler has an async ``handle``.
    - Ensures the final stage is an envelope builder (by `_produces_envelope`).
    - Optionally performs extra debug checks when GEMINI_PIPELINE_VALIDATE=1.
    """
    if not handlers:
        raise ValueError("compose_pipeline() requires at least one handler")

    for h in handlers:
        fn = getattr(h, "handle", None)
        if not callable(fn) or not inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"Handler {type(h).__name__} must define an async 'handle' method"
            )

    last = handlers[-1]
    if not bool(getattr(last, "_produces_envelope", False)):
        raise TypeError(
            "The final handler must produce a ResultEnvelope (e.g., ResultBuilder). "
            "Mark custom terminal stages with '_produces_envelope = True'."
        )

    if os.getenv("GEMINI_PIPELINE_VALIDATE") == "1":
        # Debug-only: soft checks for annotations to help extension authors
        for h in handlers:
            try:
                hints = h.handle.__annotations__
                if not hints:
                    log.warning("Handler %s lacks type annotations", type(h).__name__)
                    continue
                if "return" not in hints:
                    log.warning(
                        "Handler %s.handle missing return annotation", type(h).__name__
                    )
            except Exception as e:  # pragma: no cover - advisory only
                log.debug("Pipeline validate hints failed for %s: %s", h, e)

    return list(handlers)


__all__ = ("compose_pipeline",)
