"""Result building stage of the pipeline (minimal slice).

Transforms a FinalizedCommand (with raw API response) into the minimal
result structure expected by ConversationManager and integration tests:
{"success": True/False, "answers": [str], "metrics": {...}}
"""

from typing import Any

from gemini_batch.core.exceptions import ValidationError
from gemini_batch.core.types import Failure, FinalizedCommand, Result, Success
from gemini_batch.pipeline.base import BaseAsyncHandler


class ResultBuilder(
    BaseAsyncHandler[FinalizedCommand, dict[str, Any], ValidationError]
):
    """Builds the final result from API responses (minimal slice)."""

    def __init__(self) -> None:
        """Initialize a pure result transformation handler."""
        return

    async def handle(
        self, command: FinalizedCommand
    ) -> Result[dict[str, Any], ValidationError]:
        """Transform a finalized command into the final result dictionary."""
        try:
            raw = command.raw_api_response

            # Minimal extractor shim: prefer simple shape, tolerate one common nested shape
            def extract_text(payload: Any) -> str | None:
                if isinstance(payload, dict):
                    simple = payload.get("text")
                    if isinstance(simple, str):
                        return simple
                    # Try a conservative SDK-like nested shape without importing SDK types
                    candidates = payload.get("candidates")
                    if isinstance(candidates, list) and candidates:
                        first = candidates[0]
                        if isinstance(first, dict):
                            content = first.get("content")
                            if isinstance(content, dict):
                                parts = content.get("parts")
                                if isinstance(parts, list) and parts:
                                    part0 = parts[0]
                                    if isinstance(part0, dict):
                                        nested_text = part0.get("text")
                                        if isinstance(nested_text, str):
                                            return nested_text
                return None

            # Minimal: derive a single textual answer from the raw payload
            text = extract_text(raw)
            if text is None:
                return Failure(ValidationError("Missing text in API response"))

            # Include telemetry durations if available
            durations = {}
            try:
                durations = dict(command.telemetry_data.get("durations", {}))
            except Exception:
                durations = {}

            # Include token validation metrics if present
            token_validation = {}
            try:
                token_validation = dict(
                    command.telemetry_data.get("token_validation", {})
                )
            except Exception:
                token_validation = {}

            result: dict[str, Any] = {
                "success": True,
                "answers": [str(text)],
                "metrics": {
                    "durations": durations,
                    "token_validation": token_validation,
                },
            }
            return Success(result)
        except ValidationError as e:
            return Failure(e)
        except Exception as e:
            return Failure(ValidationError(f"Failed to build result: {e}"))
