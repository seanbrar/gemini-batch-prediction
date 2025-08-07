"""Execution planning stage of the pipeline.

This module contains the handler responsible for creating execution plans
from resolved commands, determining how to process the request efficiently.
"""

from gemini_batch.core.exceptions import ConfigurationError
from gemini_batch.core.types import PlannedCommand, ResolvedCommand
from gemini_batch.pipeline.base import BaseAsyncHandler


class ExecutionPlanner(
    BaseAsyncHandler[ResolvedCommand, PlannedCommand, ConfigurationError]
):
    """Creates execution plans from resolved commands.

    This handler analyzes the resolved sources and configuration to determine
    how to execute the request, including token estimation, caching strategy,
    and payload construction.
    """
