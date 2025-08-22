"""The primary user-facing entry point for the pipeline."""

from time import perf_counter
from typing import Any, cast

from gemini_batch.config import FrozenConfig, resolve_config
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
from gemini_batch.pipeline.rate_limit_handler import RateLimitHandler
from gemini_batch.pipeline.registries import CacheRegistry, FileRegistry
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
        config: FrozenConfig,
        pipeline_handlers: list[BaseAsyncHandler[Any, Any, GeminiBatchError]]
        | None = None,
    ):
        """Initialize the executor with configuration.

        Args:
            config: Configuration for the pipeline (FrozenConfig).
            pipeline_handlers: Optional list of handlers to override the default pipeline.
        """
        self.config = config
        self._cache_registry = CacheRegistry()
        self._file_registry = FileRegistry()
        self._pipeline = pipeline_handlers or self._build_default_pipeline(config)

    def _build_default_pipeline(
        self, config: FrozenConfig
    ) -> list[BaseAsyncHandler[Any, Any, GeminiBatchError]]:
        """Build the default pipeline of handlers.

        Returns:
            The list of handlers that comprise the default pipeline.
        """
        # Optional real adapter factory when explicitly requested
        adapter_factory = None
        use_real = config.use_real_api
        if use_real:
            # Use the provider adapter seam to get the right configuration
            from gemini_batch.pipeline.adapters.registry import build_provider_config

            _ = build_provider_config(config.provider, config)

            def _factory(api_key: str) -> Any:  # defer import until needed
                from gemini_batch.pipeline.adapters.gemini import GoogleGenAIAdapter

                return GoogleGenAIAdapter(api_key)

            adapter_factory = _factory

        handlers = [
            SourceHandler(),
            ExecutionPlanner(),
            RateLimitHandler(),
            APIHandler(
                telemetry=None,
                registries={
                    "cache": self._cache_registry,
                    "files": self._file_registry,
                },
                adapter_factory=adapter_factory,
            ),
            ResultBuilder(),
        ]
        return cast("list[BaseAsyncHandler[Any, Any, GeminiBatchError]]", handlers)

    async def execute(self, _command: InitialCommand) -> dict[str, Any]:
        """Execute a command through the pipeline.

        Args:
            _command: The initial command to execute (sources, prompts, config).

        Returns:
            A minimal result dictionary produced by the `ResultBuilder` stage.

        Raises:
            PipelineError: If any stage returns a failure result.
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
                # Avoid setdefault chaining to satisfy typing and be resilient to bad shapes
                existing = current.telemetry_data.get("durations")
                if isinstance(existing, dict):
                    existing.update(stage_durations)
                else:
                    current.telemetry_data["durations"] = dict(stage_durations)

        # ResultBuilder returns the final result dict
        if isinstance(current, dict):
            # Ensure stage durations (including ResultBuilder) are surfaced
            metrics_container = current.setdefault("metrics", {})
            if isinstance(metrics_container, dict):
                existing_durations = metrics_container.get("durations")
                if isinstance(existing_durations, dict):
                    existing_durations.update(stage_durations)
                else:
                    metrics_container["durations"] = dict(stage_durations)
            return current
        # Defensive fallback to avoid leaking internals; keep a uniform return shape.
        return {
            "success": False,
            "answers": [],
            "metrics": {},
            "error": "Unexpected pipeline state",
            "stage": last_stage_name,
        }


def create_executor(
    config: FrozenConfig | None = None,
) -> GeminiExecutor:
    """Create an executor with optional configuration.

    If no configuration is provided, it will be resolved from the environment
    using the new configuration system.

    Args:
        config: Optional configuration object (FrozenConfig or None).

    Returns:
        An instance of GeminiExecutor.
    """
    # This is the only place where ambient configuration is resolved.
    # Use the new configuration system; explain=False returns FrozenConfig
    final_config = config if config is not None else resolve_config()

    return GeminiExecutor(final_config)
