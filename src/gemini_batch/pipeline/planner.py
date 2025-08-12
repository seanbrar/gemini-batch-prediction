"""Execution planning stage of the pipeline (minimal slice).

Creates execution plans from resolved commands by assembling a neutral
text payload from prompts. Advanced features like token estimation and
caching are intentionally deferred for a subsequent PR.
"""

from __future__ import annotations

from gemini_batch.core.exceptions import ConfigurationError
from gemini_batch.core.types import (
    APICall,
    ExecutionPlan,
    Failure,
    PlannedCommand,
    ResolvedCommand,
    Result,
    Success,
    TextPart,
)
from gemini_batch.pipeline.base import BaseAsyncHandler


class ExecutionPlanner(
    BaseAsyncHandler[ResolvedCommand, PlannedCommand, ConfigurationError]
):
    """Creates execution plans from resolved commands (minimal slice)."""

    def __init__(self) -> None:
        """Initialize a stateless planner (no dependencies)."""
        return

    async def handle(
        self, command: ResolvedCommand
    ) -> Result[PlannedCommand, ConfigurationError]:
        """Create a minimal execution plan for the resolved command."""
        try:
            initial = command.initial
            model_name: str = str(initial.config.get("model") or "gemini-2.0-flash")

            # Validate the prompts exist to avoid invalid states.
            if not initial.prompts:
                return Failure(ConfigurationError("At least one prompt is required."))

            # Assemble a minimal text payload from prompts using a neutral type.
            joined_prompt = "\n\n".join(initial.prompts)
            api_call = APICall(
                model_name=model_name,
                api_parts=[TextPart(text=joined_prompt)],
                api_config={},
            )

            plan = ExecutionPlan(primary_call=api_call, fallback_call=None)
            planned = PlannedCommand(resolved=command, execution_plan=plan)
            return Success(planned)
        except ConfigurationError as e:
            return Failure(e)
        except Exception as e:  # Defensive: normalize unexpected errors
            return Failure(ConfigurationError(f"Failed to plan execution: {e}"))
