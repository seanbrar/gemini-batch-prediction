"""The primary user-facing entry point for the pipeline."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any

from gemini_batch.config import FrozenConfig, resolve_config
from gemini_batch.core.exceptions import (
    GeminiBatchError,
    InvariantViolationError,
    PipelineError,
)
from gemini_batch.core.types import (
    Failure,
    FinalizedCommand,
    InitialCommand,
    Result,
    ResultEnvelope,
    Success,
    is_result_envelope,
)
from gemini_batch.pipeline._erasure import ErasedAsyncHandler, erase
from gemini_batch.pipeline.api_handler import APIHandler
from gemini_batch.pipeline.cache_stage import CacheStage
from gemini_batch.pipeline.planner import ExecutionPlanner
from gemini_batch.pipeline.rate_limit_handler import RateLimitHandler
from gemini_batch.pipeline.registries import CacheRegistry, FileRegistry
from gemini_batch.pipeline.result_builder import ResultBuilder
from gemini_batch.pipeline.source_handler import SourceHandler
from gemini_batch.telemetry import TelemetryContext

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemini_batch.pipeline.base import BaseAsyncHandler


class GeminiExecutor:
    """Executes commands through a pipeline of handlers.

    This class manages the flow of commands through a series of processing
    stages, each responsible for a specific aspect of the request.
    """

    def __init__(
        self,
        config: FrozenConfig,
        pipeline_handlers: (
            Iterable[BaseAsyncHandler[Any, Any, GeminiBatchError]]
            | Iterable[ErasedAsyncHandler]
            | None
        ) = None,
    ):
        """Initialize the executor with configuration.

        Args:
            config: Configuration for the pipeline (FrozenConfig).
            pipeline_handlers: Optional list of handlers to override the default pipeline.
        """
        self.config = config
        self._cache_registry = CacheRegistry()
        self._file_registry = FileRegistry()
        # Build or validate a raw pipeline for tests and introspection
        raw_handlers: list[Any] = list(
            pipeline_handlers or self._build_default_pipeline(config)
        )
        if not raw_handlers:
            raise ValueError("Pipeline may not be empty; provide at least one handler.")
        # Use provided handlers as-is; executor enforces invariants at runtime.
        self._pipeline = raw_handlers
        # Internal execution pipeline (type-erased wrappers). Supports mixed inputs.
        self._exec_pipeline: list[ErasedAsyncHandler] = [
            erase(h) for h in self._pipeline
        ]

    def _build_default_pipeline(self, config: FrozenConfig) -> list[Any]:
        """Build the default pipeline of handlers.

        Returns:
            The list of handlers that comprise the default pipeline.
        """
        # Optional real adapter factory when explicitly requested
        adapter_factory = None
        if config.use_real_api:
            # Use the provider adapter seam to get the right configuration
            from gemini_batch.pipeline.adapters.registry import build_provider_config

            _ = build_provider_config(config.provider, config)

            def _factory(api_key: str) -> Any:  # defer import until needed
                from gemini_batch.pipeline.adapters.gemini import GoogleGenAIAdapter

                return GoogleGenAIAdapter(api_key)

            adapter_factory = _factory

        # Always include the RateLimitHandler. Enforcement is plan-driven:
        # the planner attaches a RateConstraint only when using the real API.
        # When no constraint is present, the handler is a no-op.
        handlers: list[Any] = [
            SourceHandler(),
            ExecutionPlanner(),
            RateLimitHandler(),
        ]

        handlers += [
            CacheStage(
                registries={
                    "cache": self._cache_registry,
                },
                adapter_factory=adapter_factory,
            ),
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
        return handlers

    async def execute(self, _command: InitialCommand) -> ResultEnvelope:
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

        for handler in self._exec_pipeline:
            last_stage_name = handler.stage_name
            with ctx("pipeline.stage", stage=last_stage_name):
                start = perf_counter()
                result: Result[Any, GeminiBatchError] = await handler.handle(current)
                duration = perf_counter() - start
            stage_durations[last_stage_name] = duration

            # Guard: handlers must return Success|Failure
            if not isinstance(result, Success | Failure):
                ctx.count("pipeline.invariant_violation", stage=last_stage_name)
                raise InvariantViolationError(
                    "Handler returned a non-Result value; expected Success|Failure.",
                    stage_name=last_stage_name,
                )

            if isinstance(result, Failure):
                # Increment operational counter, then raise with true stage identity
                ctx.count("pipeline.error", stage=last_stage_name)
                raise PipelineError(str(result.error), last_stage_name, result.error)
            current = result.value

            # Attach telemetry durations to FinalizedCommand so ResultBuilder can surface metrics
            if isinstance(current, FinalizedCommand):
                # Avoid setdefault chaining to satisfy typing and be resilient to bad shapes
                existing = current.telemetry_data.get("durations")
                if isinstance(existing, dict):
                    existing.update(stage_durations)
                else:
                    current.telemetry_data["durations"] = dict(stage_durations)

        # ResultBuilder must return the final ResultEnvelope
        if is_result_envelope(current):
            # Ensure stage durations (including ResultBuilder) are surfaced
            metrics_container = current.setdefault("metrics", {})
            if isinstance(metrics_container, dict):
                existing_durations = metrics_container.get("durations")
                if isinstance(existing_durations, dict):
                    existing_durations.update(stage_durations)
                else:
                    metrics_container["durations"] = dict(stage_durations)
            return current
        # Invariant: final stage must produce a ResultEnvelope-shaped dict
        ctx.count(
            "pipeline.invariant_violation",
            stage=last_stage_name or "unknown_stage",
        )
        raise InvariantViolationError(
            "Executor ended without a ResultEnvelope; ensure the final stage produces the envelope (e.g., ResultBuilder).",
            stage_name=last_stage_name,
        )

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Return the current pipeline's stage names in execution order."""
        return tuple(h.stage_name for h in self._exec_pipeline)

    @property
    def raw_pipeline(
        self,
    ) -> tuple[
        BaseAsyncHandler[Any, Any, GeminiBatchError] | ErasedAsyncHandler,
        ...,
    ]:
        """Return the original handlers as constructed (read-only view).

        Useful for introspection and testing. This exposes either typed handlers
        or erased handlers, depending on how the executor was initialized.
        """
        return tuple(self._pipeline)


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
