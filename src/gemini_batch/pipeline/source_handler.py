"""Source validation stage of the pipeline."""

import logging

from gemini_batch.core.exceptions import SourceError
from gemini_batch.core.types import (
    Failure,
    InitialCommand,
    ResolvedCommand,
    Result,
    Source,
    Success,
)
from gemini_batch.pipeline.base import BaseAsyncHandler

logger = logging.getLogger(__name__)


class SourceHandler(BaseAsyncHandler[InitialCommand, ResolvedCommand, SourceError]):
    """Validate that inputs are explicit `Source` objects.

    Legacy implicit detection has been removed in favor of explicit, typed
    constructors on `Source` and helpers in `core.sources`. This handler
    is intentionally minimal to keep the pipeline data-centric.
    """

    def __init__(self) -> None:  # Keep constructor trivial per simplicity rubric
        """Initialize handler (no heavy dependencies)."""
        # Intentionally empty; handler is stateless and uses stdlib utilities

    async def handle(
        self, command: InitialCommand
    ) -> Result[ResolvedCommand, SourceError]:
        """Resolve sources in a command into immutable `Source` objects."""
        try:
            # All sources must already be explicit `Source` objects.
            # This handler validates and forwards them without implicit inference.
            for s in command.sources:
                if not isinstance(s, Source):
                    return Failure(
                        SourceError(
                            "All inputs must be explicit `Source` objects. "
                            "Use `types.Source.from_text(...)` or `types.Source.from_file(...)`."
                        )
                    )
            return Success(
                ResolvedCommand(
                    initial=command, resolved_sources=tuple(command.sources)
                )
            )
        except SourceError as e:
            return Failure(e)
        except Exception as e:  # Defensive guardrail
            return Failure(SourceError(f"Failed to validate sources: {e}"))
