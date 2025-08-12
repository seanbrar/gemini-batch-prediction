"""The primary user-facing entry point for the pipeline."""

from contextlib import suppress
from time import perf_counter
from typing import Any, cast

from gemini_batch.config import GeminiConfig, get_ambient_config
from gemini_batch.core.exceptions import GeminiBatchError, PipelineError
from gemini_batch.core.types import (
    Failure,
    FinalizedCommand,
    InitialCommand,
    Result,
)
from gemini_batch.pipeline.api_handler import APIHandler
from gemini_batch.pipeline.base import BaseAsyncHandler
from gemini_batch.pipeline.planner import ExecutionPlanner
from gemini_batch.pipeline.result_builder import ResultBuilder
from gemini_batch.pipeline.source_handler import SourceHandler
from gemini_batch.telemetry import TelemetryContext


class GeminiExecutor:
    """Executes commands through a pipeline of handlers.

    This class manages the flow of commands through a series of processing
    stages, each responsible for a specific aspect of the request.
    """

    def __init__(
        self,
        config: GeminiConfig,
        pipeline_handlers: list[BaseAsyncHandler[Any, Any, GeminiBatchError]]
        | None = None,
    ):
        """Initialize the executor with configuration and optional custom handlers.

        Args:
            config: Configuration for the pipeline.
            pipeline_handlers: Optional list of handlers to override the default pipeline.
        """
        self.config = config
        self._pipeline = pipeline_handlers or self._build_default_pipeline(config)

    def _build_default_pipeline(
        self, _config: GeminiConfig
    ) -> list[BaseAsyncHandler[Any, Any, GeminiBatchError]]:
        """Build the default pipeline of handlers."""
        handlers = [SourceHandler(), ExecutionPlanner(), APIHandler(), ResultBuilder()]
        return cast("list[BaseAsyncHandler[Any, Any, GeminiBatchError]]", handlers)

    async def execute(self, _command: InitialCommand) -> dict[str, Any]:
        """Execute a command through the pipeline.

        Args:
            command: The command to execute.

        Returns:
            A dictionary containing the final result.

        Raises:
            PipelineError: If any stage in the pipeline fails.
        """
        # Sequentially run through the pipeline with explicit Result handling
        current: Any = _command
        last_stage_name = None
        stage_durations: dict[str, float] = {}
        ctx = TelemetryContext()  # no-op unless enabled with reporters/env

        for handler in self._pipeline:
            last_stage_name = handler.__class__.__name__
            with ctx("pipeline.stage", stage=last_stage_name):
                start = perf_counter()
                result: Result[Any, GeminiBatchError] = await handler.handle(current)
                duration = perf_counter() - start
            stage_durations[last_stage_name] = duration

            if isinstance(result, Failure):
                raise PipelineError(
                    str(result.error), handler.__class__.__name__, result.error
                )
            current = result.value

            # Attach telemetry durations to FinalizedCommand so ResultBuilder can surface metrics
            if isinstance(current, FinalizedCommand):
                with suppress(Exception):
                    current.telemetry_data.setdefault("durations", {}).update(
                        stage_durations
                    )

        # At this point, current should be the final result dict returned by ResultBuilder
        if isinstance(current, dict):
            # Ensure stage durations are surfaced even after the final stage
            current.setdefault("metrics", {})
            current["metrics"].setdefault("durations", {}).update(stage_durations)
            return current
        # Defensive fallback to avoid leaking internals
        return {
            "success": False,
            "answers": [],
            "metrics": {},
            "error": "Unexpected pipeline state",
            "stage": last_stage_name,
        }


def create_executor(config: GeminiConfig | None = None) -> GeminiExecutor:
    """Create an executor with optional configuration.

    If no configuration is provided, it will be resolved from the environment.
    This provides a convenient entry point for common use cases.

    Args:
        config: Optional configuration object.

    Returns:
        An instance of GeminiExecutor.
    """
    # This is the only place where ambient configuration is resolved.
    final_config = config or get_ambient_config()
    return GeminiExecutor(final_config)
