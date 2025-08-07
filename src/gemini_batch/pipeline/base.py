"""Base protocol for pipeline handlers."""

from typing import Protocol, TypeVar

from gemini_batch.core.exceptions import GeminiBatchError
from gemini_batch.core.types import Result

# Contravariant input (handlers can accept supertypes), invariant output
T_In = TypeVar("T_In", contravariant=True)
T_Out = TypeVar("T_Out")
T_Error = TypeVar("T_Error", bound=GeminiBatchError)


class BaseAsyncHandler(Protocol[T_In, T_Out, T_Error]):
    """Protocol for asynchronous pipeline handlers.

    Each handler performs a single transformation on the command object,
    making it easy to test and reason about.
    """

    async def handle(self, command: T_In) -> Result[T_Out, T_Error]:
        """Process a command object.

        Args:
            command: The input command state from the previous pipeline stage.

        Returns:
            A Result object containing either the next command state or an error.
        """
        ...
