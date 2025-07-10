from abc import ABC, abstractmethod
from typing import List


class BasePromptBuilder(ABC):
    """Abstract base class for all prompt builders."""

    @abstractmethod
    def create_prompt(self, questions: List[str]) -> str:
        """Creates the full prompt text to be sent to the model."""
        pass
