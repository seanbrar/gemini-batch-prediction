from abc import ABC, abstractmethod  # noqa: D100


class BasePromptBuilder(ABC):
    """Abstract base class for all prompt builders."""

    @abstractmethod
    def create_prompt(self, questions: list[str]) -> str:
        """Creates the full prompt text to be sent to the model."""
