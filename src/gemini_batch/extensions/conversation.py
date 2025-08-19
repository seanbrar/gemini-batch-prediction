"""High-level, stateful conversation management.

This module provides the ConversationManager for multi-turn interactions and
PersistenceHandlers for saving and loading conversation state, ensuring a
clean separation of concerns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import TYPE_CHECKING, Any

from gemini_batch.config.compatibility import ensure_frozen_config
from gemini_batch.core.types import ConversationTurn, InitialCommand

if TYPE_CHECKING:
    from pathlib import Path

    from gemini_batch.executor import GeminiExecutor


logger = logging.getLogger(__name__)

# --- Pure State Management ---


class ConversationManager:
    """A pure state manager for multi-turn conversations.

    This class holds the conversation's sources and history. It does not
    perform any processing itself; instead, it constructs a Command and
    delegates execution to an injected GeminiExecutor, embodying the principle
    of separating state from behavior.
    """

    def __init__(self, sources: list[Any], executor: GeminiExecutor):
        """Initializes the ConversationManager.

        Args:
            sources: A list of initial source materials (e.g., paths, URLs).
            executor: An initialized GeminiExecutor instance. This is a required,
                explicit dependency, ensuring the manager is decoupled from
                pipeline construction and configuration.
        """
        self.sources = tuple(sources)
        self.history: list[ConversationTurn] = []
        self._executor = executor

    async def ask(self, prompt: str) -> ConversationTurn:
        """Asks a question within the context of the conversation.

        This method constructs the command and delegates to the executor. The
        returned ConversationTurn is a rich, structured object, making the
        result explicit and robust, per our rubric analysis.

        Args:
            prompt: The user's question.

        Returns:
            A ConversationTurn object containing the question, answer, and
            metadata about the success or failure of the turn.
        """
        # 1. Construct the command object. This is the manager's primary job.
        # It packages the current state for the stateless pipeline.
        # Ensure we use FrozenConfig for new commands
        frozen_config = ensure_frozen_config(self._executor.config)
        command = InitialCommand(
            sources=self.sources,
            prompts=(prompt,),
            config=frozen_config,
            history=tuple(self.history),
        )

        # 2. Delegate to the executor. The manager is blind to the inner
        # workings of the pipeline (planning, API calls, etc.).
        result_dict = await self._executor.execute(command)

        # 3. Process the result and update internal state.
        is_error = not result_dict.get("success", False)
        answer = "Error: No answer found."
        if result_dict.get("answers"):
            # Assuming the simple case of one answer for one prompt.
            answer = result_dict["answers"][0]

        # The new turn is created from the final, processed result.
        new_turn = ConversationTurn(
            question=prompt,
            answer=answer,
            is_error=is_error,
        )
        self.history.append(new_turn)

        return new_turn

    # --- Simple source and history management methods ---

    def add_source(self, source: Any) -> None:
        """Adds a new source to the conversation's context."""
        # Note: In a real implementation, I'd want to consider how
        # this affects the tuple (e.g., create a new tuple).
        # For the sketch, a list is simpler to show the concept.
        if not isinstance(self.sources, list):
            sources_list = list(self.sources)
        if source not in sources_list:
            sources_list.append(source)
            self.sources = tuple(sources_list)  # Maintain immutability

    def get_detailed_history(self) -> tuple[ConversationTurn, ...]:
        """Returns an immutable snapshot of the detailed conversation history."""
        return tuple(self.history)

    def clear_history(self) -> None:
        """Clears the conversation history, preserving sources and config."""
        self.history.clear()


# --- Decoupled Persistence ---


class BasePersistenceHandler(ABC):
    """Abstract base class for saving and loading conversation state.

    This decouples the ConversationManager from the details of I/O,
    adhering to the single responsibility principle. The manager manages
    state; the handler manages storage.
    """

    @abstractmethod
    def save(self, manager: ConversationManager, path: Path) -> None:
        """Saves the state of a ConversationManager to a given path."""
        ...

    @abstractmethod
    def load(self, path: Path, executor: GeminiExecutor) -> ConversationManager:
        """Loads state from a path and returns a new ConversationManager."""
        ...


class JSONPersistenceHandler(BasePersistenceHandler):
    """Saves and loads conversation state to/from a JSON file."""

    def save(self, manager: ConversationManager, path: Path) -> None:
        """Saves the session state to a JSON file."""
        logger.info(f"PERSISTENCE: Saving session to {path}...")
        # In a real implementation, I would serialize the manager's
        # sources and history to JSON here.
        # This is just a stub.
        _state_to_save = {
            # Sketch of data to be saved
            "sources": [str(s) for s in manager.sources],
            "history": [
                {
                    "question": turn.question,
                    "answer": turn.answer,
                    "is_error": turn.is_error,
                }
                for turn in manager.history
            ],
        }
        # Path(path).write_text(json.dumps(state_to_save, indent=2))  # noqa: ERA001

    def load(self, path: Path, executor: GeminiExecutor) -> ConversationManager:
        """Loads a session from a JSON file."""
        logger.info(f"PERSISTENCE: Loading session from {path}...")
        # In a real implementation, I would read the JSON file,
        # reconstruct the sources and history, and create a new
        # ConversationManager instance.
        # This is just a stub.
        # loaded_state = json.loads(Path(path).read_text())  # noqa: ERA001
        # sources = loaded_state["sources"]  # noqa: ERA001
        # ... and so on ...
        mock_loaded_sources = ["loaded_source_from_json.txt"]
        manager = ConversationManager(sources=mock_loaded_sources, executor=executor)
        # ... loop to reconstruct history ...
        return manager  # noqa: RET504
