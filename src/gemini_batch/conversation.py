from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import (
    Any,
    Optional,
    Unpack,
)
from uuid import uuid4

from gemini_batch.batch_processor import BatchProcessor
from gemini_batch.client.token_counter import TokenCounter
from gemini_batch.constants import CACHING_VALIDATION_THRESHOLD, LARGE_CONTENT_THRESHOLD

from .config import ConversationConfig, GeminiConfig, ProcessorProtocol, get_config

log = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in conversation history"""

    question: str
    answer: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    sources_snapshot: list[str] = field(default_factory=list)
    cache_info: dict[str, Any] | None = None
    error: str | None = None


class ConversationSession:
    """A zero-ceremony, easy-to-use conversation session."""

    def __init__(
        self,
        sources: str | Path | list[str | Path],
        _processor: ProcessorProtocol | None = None,
        **config: Unpack[ConversationConfig],
    ):
        """Creates a conversation session.

        Examples:
            session = ConversationSession("document.pdf")
            session = ConversationSession("docs/", max_history_turns=3)
            session = ConversationSession("docs/", _processor=shared_processor)
        """
        self.sources = sources if isinstance(sources, list) else [sources]
        self.max_history_turns = config.get("max_history_turns", 5)
        self.history: list[ConversationTurn] = []
        self.session_id = str(uuid4())

        if _processor and config:
            log.warning("Ignoring config kwargs when a custom _processor is provided.")

        if _processor:
            self.processor = _processor
        else:
            # Extract only the GeminiConfig keys for the BatchProcessor
            processor_config = {
                k: v for k, v in config.items() if k in GeminiConfig.__annotations__
            }
            self.processor = BatchProcessor(**processor_config)

    def ask(self, question: str, **options) -> str:
        """Ask single question with full conversation context"""
        try:
            # Pass raw sources to BatchProcessor for proper file handling
            # Add conversation history as additional context
            history_context = self._build_history_context()

            # Prepare options with system instruction if history exists
            processor_options = options.copy()
            if history_context:
                # Fix: Merge system instruction instead of overwriting
                existing = processor_options.get("system_instruction", "")
                combined = (
                    f"{existing}\n\n{history_context}" if existing else history_context
                )
                processor_options["system_instruction"] = combined

            # Delegate to BatchProcessor for actual processing
            result = self.processor.process_questions(
                content=self.sources,  # Pass raw sources directly
                questions=[question],
                **processor_options,
            )

            # Extract answer and record successful turn
            try:
                answer = result["answers"][0]
                self._record_successful_turn(question, answer, result)
            except IndexError:
                error_message = f"No answer found for question: '{question}'"
                self._record_failed_turn(question, error_message)
                raise ValueError(error_message)

            return answer

        except Exception as e:
            self._record_failed_turn(question, str(e))
            raise e

    def ask_multiple(self, questions: list[str], **options) -> list[str]:
        """Ask batch of questions with conversation context"""
        log.debug(
            "Processing %d questions with %d sources",
            len(questions),
            len(self.sources),
        )

        try:
            # Pass raw sources to BatchProcessor for proper file handling
            # Add conversation history as additional context
            history_context = self._build_history_context()

            # Prepare options with system instruction if history exists
            processor_options = options.copy()
            if history_context:
                # Fix: Merge system instruction instead of overwriting
                existing = processor_options.get("system_instruction", "")
                combined = (
                    f"{existing}\n\n{history_context}" if existing else history_context
                )
                processor_options["system_instruction"] = combined

            # Delegate to BatchProcessor
            result = self.processor.process_questions(
                content=self.sources,  # Pass raw sources directly
                questions=questions,
                **processor_options,
            )

            # Record all successful turns
            answers = result["answers"]
            if len(answers) != len(questions):
                # TODO: Gemini response formatting is unpredictable without explicit schemas
                # See: conversation_demo.py failures with "Received 1 answers for 3 questions"
                # Root cause: Gemini sometimes returns plain text instead of structured JSON
                # Planned fix: Implement consistent response schemas in BatchProcessor refactor
                error_message = (
                    f"Received {len(answers)} answers for {len(questions)} questions."
                )
                for q in questions:
                    self._record_failed_turn(q, error_message)
                raise ValueError(error_message)

            for question, answer in zip(questions, answers, strict=False):
                self._record_successful_turn(question, answer, result)

            return answers

        except Exception as e:
            log.error("Conversation processing failed: %s", e, exc_info=True)
            for question in questions:
                self._record_failed_turn(question, str(e))
            raise e

    def _build_history_context(self, max_tokens: int | None = None) -> str | None:
        """Builds context from recent, successful conversation history with intelligent token management.

        Uses ambient configuration to determine optimal context limits and proper token counting
        when available through the processor's client.
        """
        if not self.history:
            return None

        successful_history = [turn for turn in self.history if turn.error is None]
        if not successful_history:
            return None

        # Determine max_tokens using configuration hierarchy
        if max_tokens is None:
            max_tokens = self._get_optimal_context_limit()

        # Use proper token counting if available, fallback to estimation
        token_counter = self._get_token_counter()

        # Start with most recent and work backwards until we hit token limit
        selected_turns = []
        estimated_tokens = 0

        for turn in reversed(successful_history[-self.max_history_turns :]):
            if token_counter:
                # Use proper token counting
                turn_content = f"Q: {turn.question}\nA: {turn.answer}"
                turn_tokens = self._count_tokens_with_fallback(
                    token_counter,
                    turn_content,
                )
            else:
                # Fallback to conservative estimation
                turn_tokens = max((len(turn.question) + len(turn.answer)) // 4 + 10, 50)

            if estimated_tokens + turn_tokens > max_tokens and selected_turns:
                break

            selected_turns.insert(0, turn)
            estimated_tokens += turn_tokens

        if not selected_turns:
            return None

        history_parts = []
        for i, turn in enumerate(selected_turns, 1):
            history_parts.append(f"Previous Q{i}: {turn.question}")
            history_parts.append(f"Previous A{i}: {turn.answer}")

        return "Conversation History:\n" + "\n".join(history_parts)

    def _get_optimal_context_limit(self) -> int:
        """Get optimal context limit using ambient configuration and constants."""
        try:
            # Access ambient configuration
            config = get_config()

            model_limits = config.get_model_limits(config.model)
            if model_limits:
                # Use a conservative portion of the model's context window for history
                # Reserve space for the main content and new questions
                return min(
                    model_limits.context_window // 4,
                    CACHING_VALIDATION_THRESHOLD,
                )

        except Exception:
            # Fallback to safe constant-based limit
            pass

        # Default to a reasonable portion of large content threshold
        return LARGE_CONTENT_THRESHOLD // 2  # 25k tokens as fallback

    def _get_token_counter(self) -> Optional["TokenCounter"]:
        """Get token counter from processor if available."""
        try:
            # Follow your established pattern of accessing client components
            if hasattr(self.processor, "client") and hasattr(
                self.processor.client,
                "token_counter",
            ):
                return self.processor.client.token_counter
        except (AttributeError, TypeError):
            pass
        return None

    def _count_tokens_with_fallback(
        self,
        token_counter: "TokenCounter",
        content: str,
    ) -> int:
        """Count tokens using TokenCounter with graceful fallback."""
        try:
            # Use the TokenCounter's estimation method
            config = get_config()
            estimate = token_counter.estimate_for_caching(config.model, content)
            return estimate.get("tokens", 0)
        except Exception:
            # Fallback to conservative estimation
            return max(len(content) // 4 + 10, 50)

    def _record_successful_turn(
        self,
        question: str,
        answer: str,
        result: dict[str, Any],
    ):
        """Record a successful conversation turn"""
        turn = ConversationTurn(
            question=question,
            answer=answer,
            sources_snapshot=self.sources.copy(),
            cache_info=result.get("metrics", {}).get("batch", {}),
        )
        self.history.append(turn)

    def _record_failed_turn(self, question: str, error_msg: str):
        """Record a failed conversation turn"""
        turn = ConversationTurn(
            question=question,
            answer="",
            sources_snapshot=self.sources.copy(),
            error=error_msg,
        )
        self.history.append(turn)

    # Source management
    def add_source(self, source) -> None:
        """Add new content source to conversation"""
        if source not in self.sources:
            self.sources.append(source)

    def remove_source(self, source) -> None:
        """Remove content source from conversation"""
        if source in self.sources:
            self.sources.remove(source)

    def list_sources(self) -> list[str]:
        """List current conversation sources"""
        return self.sources.copy()

    # Session persistence
    def save(self, path: str | None = None) -> str:
        """Save conversation session to file"""
        # Fix: Convert sources to strings for JSON serialization
        serializable_sources = [str(s) for s in self.sources]

        session_data = {
            "session_id": self.session_id,
            "sources": serializable_sources,
            "history": [
                {
                    "question": turn.question,
                    "answer": turn.answer,
                    "timestamp": turn.timestamp.isoformat(),
                    "sources_snapshot": [str(s) for s in turn.sources_snapshot],
                    "cache_info": turn.cache_info,
                    "error": turn.error,
                }
                for turn in self.history
            ],
            "created_at": datetime.now(UTC).isoformat(),
        }

        if path is None:
            path = f"conversation_{self.session_id}.json"

        with open(path, "w") as f:
            json.dump(session_data, f, indent=2)

        return self.session_id

    @classmethod
    def load(
        cls,
        session_id: str,
        path: str | None = None,
        processor: BatchProcessor | None = None,
    ) -> "ConversationSession":
        """Load conversation session from file"""
        if path is None:
            path = f"conversation_{session_id}.json"

        with open(path) as f:
            session_data = json.load(f)

        # Create session with restored state
        session = cls(session_data["sources"], processor)
        session.session_id = session_data["session_id"]

        # Restore conversation history
        for turn_data in session_data["history"]:
            turn = ConversationTurn(
                question=turn_data["question"],
                answer=turn_data["answer"],
                timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                sources_snapshot=turn_data["sources_snapshot"],
                cache_info=turn_data.get("cache_info"),
                error=turn_data.get("error"),
            )
            session.history.append(turn)

        return session

    # Session analytics
    def get_history(self) -> list[tuple[str, str]]:
        """Get conversation history as (question, answer) pairs"""
        return [(turn.question, turn.answer) for turn in self.history]

    def get_detailed_history(self) -> list[ConversationTurn]:
        """Get full conversation history with metadata"""
        return self.history.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive conversation statistics"""
        total_turns = len(self.history)
        successful_turns = len([t for t in self.history if t.error is None])
        error_turns = total_turns - successful_turns

        # Calculate cache efficiency
        cache_hits = sum(
            1
            for turn in self.history
            if turn.cache_info and turn.cache_info.get("cache_hit_ratio", 0) > 0
        )

        return {
            "session_id": self.session_id,
            "total_turns": total_turns,
            "successful_turns": successful_turns,
            "error_turns": error_turns,
            "success_rate": successful_turns / total_turns if total_turns > 0 else 0,
            "active_sources": len(self.sources),
            "cache_efficiency": cache_hits / total_turns if total_turns > 0 else 0,
            "session_duration": (
                self.history[-1].timestamp - self.history[0].timestamp
            ).total_seconds()
            if self.history
            else 0,
        }

    def clear_history(self) -> None:
        """Clear conversation history while preserving sources"""
        self.history.clear()


def create_conversation(
    sources: str | Path | list[str | Path],
    **config: Unpack[ConversationConfig],
) -> "ConversationSession":
    """Factory function to create a new conversation session."""
    return ConversationSession(sources, **config)


def load_conversation(
    session_id: str,
    path: str | None = None,
    **config: Unpack[ConversationConfig],
) -> "ConversationSession":
    """Factory function to load an existing conversation session."""
    # Create a processor with any specified overrides
    processor_config = {
        k: v for k, v in config.items() if k in GeminiConfig.__annotations__
    }
    processor = BatchProcessor(**processor_config)

    # The actual loading logic is preserved
    if path is None:
        path = f"conversation_{session_id}.json"

    with open(path) as f:
        session_data = json.load(f)

    # Create session, now passing the configured processor
    session = ConversationSession(
        session_data["sources"],
        _processor=processor,
        **config,
    )
    session.session_id = session_data["session_id"]

    # Restore history (logic unchanged)
    for turn_data in session_data["history"]:
        turn = ConversationTurn(
            question=turn_data["question"],
            answer=turn_data["answer"],
            timestamp=datetime.fromisoformat(turn_data["timestamp"]),
            sources_snapshot=turn_data["sources_snapshot"],
            cache_info=turn_data.get("cache_info"),
            error=turn_data.get("error"),
        )
        session.history.append(turn)

    return session
