"""Public type aliases and re-exports for end-users.

This module provides a clear, stable surface for common data types without
exposing internal pipeline mechanics.

Example:
    ```python
    from gemini_batch import types

    # Create sources using ergonomic constructors
    source = types.Source.from_file("document.pdf")
    text_source = types.Source.from_text("Raw content here")

    # Build commands with validation
    cmd = types.InitialCommand.strict(
        sources=(source,), prompts=("Analyze this document",), config=config
    )
    ```

See Also:
    For convenience functions, use `run_simple()` and `run_batch()` from the main module.
    For advanced pipeline construction, see the individual type documentation.
"""

from __future__ import annotations

from gemini_batch.core.types import (
    APICall,
    ConversationTurn,
    ExecutionPlan,
    Failure,
    FinalizedCommand,
    InitialCommand,
    PlannedCommand,
    ResolvedCommand,
    Result,
    ResultEnvelope,
    Source,
    Success,
    TokenEstimate,
)
from gemini_batch.pipeline.hints import (
    CacheHint,
    ResultHint,
)

__all__ = [  # noqa: RUF022
    # Common data types
    "ConversationTurn",
    "InitialCommand",
    "ResultEnvelope",
    "Source",
    "TokenEstimate",
    # Pipeline execution types (for power users)
    "APICall",
    "ExecutionPlan",
    "PlannedCommand",
    "ResolvedCommand",
    "FinalizedCommand",
    # Result handling
    "Result",
    "Success",
    "Failure",
    # Hint system (for extension authors)
    "CacheHint",
    "ResultHint",
]
