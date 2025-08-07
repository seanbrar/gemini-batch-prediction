"""Result building stage of the pipeline.

This module contains the handler responsible for transforming raw API responses
into user-facing results, including output parsing and validation.
"""

from typing import Any

from gemini_batch.core.exceptions import ValidationError
from gemini_batch.core.types import FinalizedCommand
from gemini_batch.pipeline.base import BaseAsyncHandler


class ResultBuilder(
    BaseAsyncHandler[FinalizedCommand, dict[str, Any], ValidationError]
):
    """Builds the final result from API responses.

    This handler transforms raw API responses into user-facing results,
    including parsing output, validation, and efficiency metrics.
    """
