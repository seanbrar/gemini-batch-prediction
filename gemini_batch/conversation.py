from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from gemini_batch import BatchProcessor


@dataclass
class ConversationTurn:
    """Single turn in conversation history"""

    question: str
    answer: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sources_snapshot: List[str] = field(default_factory=list)
    cache_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ConversationSession:
    def __init__(
        self,
        sources,
        processor: Optional[BatchProcessor] = None,
        client: Optional[Any] = None,
        max_history_turns: int = 5,
    ):
        """Initialize conversation session."""
        self.sources = sources if isinstance(sources, list) else [sources]
        self.history: List[ConversationTurn] = []
        self.session_id = str(uuid4())
        self.metadata: Dict[str, Any] = {}
        self.max_history_turns = max_history_turns

        # Create or use provided processor (dependency injection)
        self.processor = (
            processor if processor is not None else BatchProcessor(client=client)
        )

    def ask(self, question: str, **options) -> str:
        """Ask single question with full conversation context"""
        try:
            # Pass raw sources to BatchProcessor for proper file handling
            # Add conversation history as additional context
            history_context = self._build_history_context()

            # Prepare options with system instruction if history exists
            processor_options = options.copy()
            if history_context:
                processor_options["system_instruction"] = history_context

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

    def ask_multiple(self, questions: List[str], **options) -> List[str]:
        """Ask batch of questions with conversation context"""
        try:
            # Pass raw sources to BatchProcessor for proper file handling
            # Add conversation history as additional context
            history_context = self._build_history_context()

            # Prepare options with system instruction if history exists
            processor_options = options.copy()
            if history_context:
                processor_options["system_instruction"] = history_context

            # Delegate to BatchProcessor
            result = self.processor.process_questions(
                content=self.sources,  # Pass raw sources directly
                questions=questions,
                **processor_options,
            )

            # Record all successful turns
            answers = result["answers"]
            if len(answers) != len(questions):
                error_message = (
                    f"Received {len(answers)} answers for {len(questions)} questions."
                )
                for q in questions:
                    self._record_failed_turn(q, error_message)
                raise ValueError(error_message)

            for question, answer in zip(questions, answers):
                self._record_successful_turn(question, answer, result)

            return answers

        except Exception as e:
            for question in questions:
                self._record_failed_turn(question, str(e))
            raise e

    def _build_history_context(self) -> Optional[str]:
        """Builds context from recent, successful conversation history."""
        if not self.history:
            return None

        # Only include successful turns
        successful_history = [turn for turn in self.history if turn.error is None]

        # Use recent history to stay within context limits
        recent_history = successful_history[-self.max_history_turns :]

        if not recent_history:
            return None

        history_parts = []
        for i, turn in enumerate(recent_history, 1):
            history_parts.append(f"Previous Q{i}: {turn.question}")
            history_parts.append(f"Previous A{i}: {turn.answer}")

        return "Conversation History:\n" + "\n".join(history_parts)

    def _record_successful_turn(
        self,
        question: str,
        answer: str,
        result: Dict[str, Any],
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

    def list_sources(self) -> List[str]:
        """List current conversation sources"""
        return self.sources.copy()

    # Session persistence
    def save(self, path: Optional[str] = None) -> str:
        """Save conversation session to file"""
        session_data = {
            "session_id": self.session_id,
            "sources": self.sources,
            "history": [
                {
                    "question": turn.question,
                    "answer": turn.answer,
                    "timestamp": turn.timestamp.isoformat(),
                    "sources_snapshot": turn.sources_snapshot,
                    "cache_info": turn.cache_info,
                    "error": turn.error,
                }
                for turn in self.history
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
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
        path: Optional[str] = None,
        processor: Optional[BatchProcessor] = None,
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
    def get_history(self) -> List[Tuple[str, str]]:
        """Get conversation history as (question, answer) pairs"""
        return [(turn.question, turn.answer) for turn in self.history]

    def get_detailed_history(self) -> List[ConversationTurn]:
        """Get full conversation history with metadata"""
        return self.history.copy()

    def get_stats(self) -> Dict[str, Any]:
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


def create_conversation(sources, **kwargs) -> "ConversationSession":
    """Factory function to create a new conversation session."""
    processor = BatchProcessor(**kwargs)
    return ConversationSession(sources, processor)


def load_conversation(
    session_id: str, path: Optional[str] = None, **processor_options
) -> ConversationSession:
    """Factory function to load an existing conversation session."""
    processor = BatchProcessor(**processor_options)
    return ConversationSession.load(session_id, path, processor)
