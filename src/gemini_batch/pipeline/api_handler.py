"""API handling stage of the pipeline (minimal slice).

This module provides a minimal API handler that adapts a PlannedCommand into a
FinalizedCommand without performing a real network call. It serves to wire the
executor and contract-first tests. The real implementation will integrate with
the google-genai SDK and map errors to APIError.
"""

from typing import Any

from gemini_batch.core.exceptions import APIError
from gemini_batch.core.types import (
    Failure,
    FinalizedCommand,
    PlannedCommand,
    Result,
    Success,
)
from gemini_batch.pipeline.base import BaseAsyncHandler
from gemini_batch.telemetry import TelemetryContext, TelemetryContextProtocol


class APIHandler(BaseAsyncHandler[PlannedCommand, FinalizedCommand, APIError]):
    """Executes API calls according to the execution plan (placeholder).

    For now, it simulates a successful API call by attaching a mock response
    that echoes the first prompt. This keeps the pipeline deterministic and
    side-effect free for early integration tests.
    """

    def __init__(self, telemetry: TelemetryContextProtocol | None = None) -> None:
        """Initialize a thin API execution handler with optional telemetry."""
        self._telemetry: TelemetryContextProtocol = telemetry or TelemetryContext()

    async def handle(
        self, command: PlannedCommand
    ) -> Result[FinalizedCommand, APIError]:
        """Handle the planned command and return a finalized command."""
        try:
            # Enforce that a valid plan is present and has parts to execute
            plan = command.execution_plan
            primary = plan.primary_call
            if not primary.api_parts:
                return Failure(APIError("Execution plan has no parts to execute"))

            # Minimal behavior: fabricate a raw response based on the planned parts
            first_text = primary.api_parts[0].text

            # Derive a simple, deterministic total token usage simulation using
            # planner-provided TokenEstimate when available, otherwise fall back
            # to a simple prompt-based heuristic.
            estimate = command.token_estimate
            if estimate is not None:
                prompt_tokens = max(len(first_text) // 4 + 10, 0)
                # Attribute remaining to sources to improve validation fidelity
                source_tokens = max(estimate.expected_tokens - prompt_tokens, 0)
                total_tokens = prompt_tokens + source_tokens
            else:
                prompt_tokens = len(first_text) // 4 + 10
                source_tokens = 0
                total_tokens = prompt_tokens

            with self._telemetry("api.execute", model=primary.model_name):
                raw_response: dict[str, Any] = {
                    "mock": True,
                    "model": primary.model_name,
                    "text": f"echo: {first_text}",
                    "usage": {
                        "prompt_token_count": prompt_tokens,
                        "source_token_count": source_tokens,
                        "total_token_count": total_tokens,
                    },
                }
            finalized = FinalizedCommand(planned=command, raw_api_response=raw_response)
            # Attach simple validation metric if an estimate exists
            if estimate and isinstance(finalized.telemetry_data, dict):
                usage = raw_response.get("usage", {})
                actual = int(usage.get("total_token_count", 0))
                finalized.telemetry_data.setdefault("token_validation", {}).update(
                    {
                        "estimated_expected": estimate.expected_tokens,
                        "estimated_min": estimate.min_tokens,
                        "estimated_max": estimate.max_tokens,
                        "actual": actual,
                        "in_range": estimate.min_tokens
                        <= actual
                        <= estimate.max_tokens,
                    }
                )
            return Success(finalized)
        except APIError as e:
            return Failure(e)
        except Exception as e:
            return Failure(APIError(f"API handler failed: {e}"))
