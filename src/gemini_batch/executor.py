"""The primary user-facing entry point for the pipeline."""

from typing import Any

from gemini_batch.config import GeminiConfig, get_ambient_config
from gemini_batch.core.exceptions import GeminiBatchError
from gemini_batch.core.types import InitialCommand
from gemini_batch.pipeline.base import BaseAsyncHandler


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
        # TODO: Implement concrete handler classes
        # For now, return empty list to avoid abstract class instantiation
        return []
        # The APIHandler is the only one that needs specific config (the API key).
        # Others are stateless and require no initialization.
        # return [
        #     SourceHandler(),  # noqa: ERA001
        #     ExecutionPlanner(),  # noqa: ERA001
        #     APIHandler(),  # TODO: Add proper initialization  # noqa: ERA001
        #     ResultBuilder(),  # noqa: ERA001
        # ]  # noqa: ERA001, RUF100

    async def execute(self, _command: InitialCommand) -> dict[str, Any]:
        """Execute a command through the pipeline.

        Args:
            command: The command to execute.

        Returns:
            A dictionary containing the final result.

        Raises:
            PipelineError: If any stage in the pipeline fails.
        """
        # TODO: Implement pipeline execution
        # For now, return a placeholder result
        return {
            "status": "not_implemented",
            "message": "Pipeline execution not yet implemented",
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
