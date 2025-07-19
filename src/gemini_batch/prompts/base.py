from abc import ABC, abstractmethod  # noqa: D100
from typing import List  # noqa: UP035


class BasePromptBuilder(ABC):
    """Abstract base class for all prompt builders."""

    @abstractmethod
    def create_prompt(self, questions: List[str]) -> str:  # noqa: UP006
        """Creates the full prompt text to be sent to the model."""
        pass  # noqa: PIE790
