"""API handling stage of the pipeline.

This module contains the handler responsible for executing API calls
according to the execution plan, managing rate limiting, file uploads,
and API communication.
"""

from gemini_batch.core.exceptions import APIError
from gemini_batch.core.types import FinalizedCommand, PlannedCommand
from gemini_batch.pipeline.base import BaseAsyncHandler


class APIHandler(BaseAsyncHandler[PlannedCommand, FinalizedCommand, APIError]):
    """Executes API calls according to the execution plan.

    This handler manages rate limiting, file uploads, and API communication.
    It's the only component that directly communicates with the Gemini API.
    """
