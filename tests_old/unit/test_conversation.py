"""
Unit tests for conversation module components.
"""

import json
from pathlib import Path
import tempfile
from unittest.mock import Mock

from gemini_batch.conversation import ConversationSession, ConversationTurn


class TestConversationSession:
    """Test ConversationSession core functionality"""

    def test_initialization(self):
        """Test basic session initialization"""
        sources = ["test_content.txt", "data.pdf"]
        mock_processor = Mock()
        session = ConversationSession(sources, processor=mock_processor)

        assert session.sources == sources
        assert len(session.history) == 0
        assert session.max_history_turns == 5
        assert session.processor == mock_processor

    def test_initialization_with_single_source(self):
        """Test initialization with single source (non-list)"""
        source = "single_file.txt"
        mock_processor = Mock()
        session = ConversationSession(source, processor=mock_processor)

        assert session.sources == [source]

    def test_initialization_with_processor(self):
        """Test initialization with custom processor"""
        mock_processor = Mock()
        sources = ["test.txt"]
        session = ConversationSession(sources, processor=mock_processor)

        assert session.processor == mock_processor

    def test_initialization_with_client(self):
        """Test initialization with custom client"""
        mock_client = Mock()
        sources = ["test.txt"]
        session = ConversationSession(sources, client=mock_client)

        # Should create a new BatchProcessor with the client
        assert session.processor is not None


class TestSystemInstructionMerging:
    """Test the system instruction collision fix"""

    def test_merges_existing_system_instruction(self):
        """Test that existing system instruction is preserved and merged"""
        mock_processor = Mock()
        # Configure mock to return expected structure
        mock_processor.process_questions.return_value = {"answers": ["Test answer"]}
        session = ConversationSession(["test.txt"], processor=mock_processor)

        # Add some conversation history
        session.history.append(
            ConversationTurn(
                question="What is AI?",
                answer="Artificial Intelligence is a field of computer science.",
            )
        )

        # Call ask with existing system instruction
        session.ask("Tell me more", system_instruction="You are a helpful assistant.")

        # Verify the processor was called with merged system instruction
        mock_processor.process_questions.assert_called_once()
        call_args = mock_processor.process_questions.call_args
        passed_options = call_args[1]  # kwargs

        expected_system_instruction = (
            "You are a helpful assistant.\n\n"
            "Conversation History:\n"
            "Previous Q1: What is AI?\n"
            "Previous A1: Artificial Intelligence is a field of computer science."
        )

        assert passed_options["system_instruction"] == expected_system_instruction

    def test_uses_history_when_no_existing_system_instruction(self):
        """Test that history is used as system instruction when none exists"""
        mock_processor = Mock()
        # Configure mock to return expected structure
        mock_processor.process_questions.return_value = {"answers": ["Test answer"]}
        session = ConversationSession(["test.txt"], processor=mock_processor)

        # Add conversation history
        session.history.append(
            ConversationTurn(
                question="What is AI?",
                answer="Artificial Intelligence is a field of computer science.",
            )
        )

        # Call ask without system instruction
        session.ask("Tell me more")

        # Verify the processor was called with history as system instruction
        mock_processor.process_questions.assert_called_once()
        call_args = mock_processor.process_questions.call_args
        passed_options = call_args[1]  # kwargs

        expected_system_instruction = (
            "Conversation History:\n"
            "Previous Q1: What is AI?\n"
            "Previous A1: Artificial Intelligence is a field of computer science."
        )

        assert passed_options["system_instruction"] == expected_system_instruction

    def test_no_system_instruction_when_no_history(self):
        """Test that no system instruction is added when no history exists"""
        mock_processor = Mock()
        # Configure mock to return expected structure
        mock_processor.process_questions.return_value = {"answers": ["Test answer"]}
        session = ConversationSession(["test.txt"], processor=mock_processor)

        # Call ask without any history
        session.ask("What is AI?")

        # Verify the processor was called without system instruction
        mock_processor.process_questions.assert_called_once()
        call_args = mock_processor.process_questions.call_args
        passed_options = call_args[1]  # kwargs

        assert "system_instruction" not in passed_options

    def test_merges_multiple_questions_with_system_instruction(self):
        """Test system instruction merging with ask_multiple"""
        mock_processor = Mock()
        # Configure mock to return expected structure
        mock_processor.process_questions.return_value = {
            "answers": ["Answer 1", "Answer 2"]
        }
        session = ConversationSession(["test.txt"], processor=mock_processor)

        # Add conversation history
        session.history.append(
            ConversationTurn(
                question="What is AI?",
                answer="Artificial Intelligence is a field of computer science.",
            )
        )

        # Call ask_multiple with existing system instruction
        questions = ["Tell me more", "Give examples"]
        session.ask_multiple(
            questions, system_instruction="You are a helpful assistant."
        )

        # Verify the processor was called with merged system instruction
        mock_processor.process_questions.assert_called_once()
        call_args = mock_processor.process_questions.call_args
        passed_options = call_args[1]  # kwargs

        expected_system_instruction = (
            "You are a helpful assistant.\n\n"
            "Conversation History:\n"
            "Previous Q1: What is AI?\n"
            "Previous A1: Artificial Intelligence is a field of computer science."
        )

        assert passed_options["system_instruction"] == expected_system_instruction


class TestJSONSerialization:
    """Test the JSON serialization fix for Path objects"""

    def test_save_converts_path_objects_to_strings(self):
        """Test that Path objects are converted to strings during save"""
        # Create a session with Path objects
        sources = [Path("test_file.txt"), Path("data/document.pdf")]
        mock_processor = Mock()
        session = ConversationSession(sources, processor=mock_processor)

        # Add some history with Path objects in sources_snapshot
        session.history.append(
            ConversationTurn(
                question="What is this?",
                answer="A test file",
                sources_snapshot=[Path("test_file.txt"), Path("data/document.pdf")],
            )
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            session.save(temp_path)

            # Load the saved file and verify it's valid JSON
            with open(temp_path) as f:
                saved_data = json.load(f)

            # Verify sources are strings
            assert all(isinstance(s, str) for s in saved_data["sources"])
            assert saved_data["sources"] == ["test_file.txt", "data/document.pdf"]

            # Verify sources_snapshot in history are strings
            history_entry = saved_data["history"][0]
            assert all(isinstance(s, str) for s in history_entry["sources_snapshot"])
            assert history_entry["sources_snapshot"] == [
                "test_file.txt",
                "data/document.pdf",
            ]

        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)

    def test_save_handles_mixed_source_types(self):
        """Test that save handles mixed string and Path sources"""
        sources = ["string_source.txt", Path("path_source.pdf"), "another_string.md"]
        mock_processor = Mock()
        session = ConversationSession(sources, processor=mock_processor)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            session.save(temp_path)

            with open(temp_path) as f:
                saved_data = json.load(f)

            # All sources should be strings
            assert all(isinstance(s, str) for s in saved_data["sources"])
            assert saved_data["sources"] == [
                "string_source.txt",
                "path_source.pdf",
                "another_string.md",
            ]

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_restores_session_correctly(self):
        """Test that load restores session with string sources"""
        # Create original session
        original_sources = [Path("test_file.txt"), "string_source.pdf"]
        mock_processor = Mock()
        # Configure mock to return expected structure
        mock_processor.process_questions.return_value = {"answers": ["Test answer"]}
        session = ConversationSession(original_sources, processor=mock_processor)

        # Add some history
        session.ask("What is this?")  # This will fail but still record the turn

        # Save and load
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            session.save(temp_path)

            # Load the session
            loaded_session = ConversationSession.load(
                session.session_id, temp_path, processor=mock_processor
            )

            # Verify sources are restored as strings
            assert all(isinstance(s, str) for s in loaded_session.sources)
            assert loaded_session.sources == ["test_file.txt", "string_source.pdf"]

            # Verify session ID is preserved
            assert loaded_session.session_id == session.session_id

            # Verify history is restored
            assert len(loaded_session.history) == len(session.history)

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestConversationHistory:
    """Test conversation history management"""

    def test_build_history_context_with_successful_turns(self):
        """Test that only successful turns are included in history context"""
        mock_processor = Mock()
        session = ConversationSession(["test.txt"], processor=mock_processor)

        # Add successful and failed turns
        session.history.append(
            ConversationTurn(
                question="What is AI?", answer="Artificial Intelligence", error=None
            )
        )
        session.history.append(
            ConversationTurn(
                question="What is ML?", answer="", error="Failed to process"
            )
        )
        session.history.append(
            ConversationTurn(
                question="What is NLP?",
                answer="Natural Language Processing",
                error=None,
            )
        )

        context = session._build_history_context()

        # Should only include successful turns
        assert "What is AI?" in context
        assert "What is ML?" not in context  # Failed turn
        assert "What is NLP?" in context

    def test_history_context_respects_max_turns(self):
        """Test that history context respects max_history_turns limit"""
        mock_processor = Mock()
        session = ConversationSession(
            ["test.txt"], max_history_turns=2, processor=mock_processor
        )

        # Add more turns than the limit
        for i in range(5):
            session.history.append(
                ConversationTurn(
                    question=f"Question {i}?", answer=f"Answer {i}", error=None
                )
            )

        context = session._build_history_context()

        # Should only include the last 2 turns
        assert "Question 3?" in context
        assert "Question 4?" in context
        assert "Question 0?" not in context
        assert "Question 1?" not in context
        assert "Question 2?" not in context

    def test_no_history_context_when_no_successful_turns(self):
        """Test that no context is built when all turns failed"""
        mock_processor = Mock()
        session = ConversationSession(["test.txt"], processor=mock_processor)

        # Add only failed turns
        session.history.append(
            ConversationTurn(
                question="What is AI?", answer="", error="Failed to process"
            )
        )

        context = session._build_history_context()
        assert context is None


class TestSourceManagement:
    """Test source management functionality"""

    def test_add_source(self):
        """Test adding a new source"""
        mock_processor = Mock()
        session = ConversationSession(["existing.txt"], processor=mock_processor)

        session.add_source("new_source.pdf")

        assert "new_source.pdf" in session.sources
        assert len(session.sources) == 2

    def test_add_source_prevents_duplicates(self):
        """Test that adding duplicate sources is prevented"""
        mock_processor = Mock()
        session = ConversationSession(["existing.txt"], processor=mock_processor)

        session.add_source("existing.txt")  # Try to add duplicate

        assert session.sources == ["existing.txt"]  # No change

    def test_remove_source(self):
        """Test removing a source"""
        mock_processor = Mock()
        session = ConversationSession(
            ["file1.txt", "file2.pdf"], processor=mock_processor
        )

        session.remove_source("file1.txt")

        assert "file1.txt" not in session.sources
        assert "file2.pdf" in session.sources
        assert len(session.sources) == 1

    def test_remove_nonexistent_source(self):
        """Test removing a source that doesn't exist"""
        mock_processor = Mock()
        session = ConversationSession(["file1.txt"], processor=mock_processor)

        session.remove_source("nonexistent.txt")

        assert session.sources == ["file1.txt"]  # No change

    def test_list_sources(self):
        """Test listing sources"""
        sources = ["file1.txt", "file2.pdf", "file3.md"]
        mock_processor = Mock()
        session = ConversationSession(sources, processor=mock_processor)

        listed_sources = session.list_sources()

        assert listed_sources == sources
        # Should be a copy, not the same list
        assert listed_sources is not session.sources


class TestSessionAnalytics:
    """Test session analytics and statistics"""

    def test_get_stats_with_mixed_history(self):
        """Test statistics calculation with successful and failed turns"""
        mock_processor = Mock()
        session = ConversationSession(["test.txt"], processor=mock_processor)

        # Add some history
        session.history.append(ConversationTurn(question="Q1", answer="A1", error=None))
        session.history.append(
            ConversationTurn(question="Q2", answer="", error="Failed")
        )
        session.history.append(ConversationTurn(question="Q3", answer="A3", error=None))

        stats = session.get_stats()

        assert stats["total_turns"] == 3
        assert stats["successful_turns"] == 2
        assert stats["error_turns"] == 1
        assert stats["success_rate"] == 2 / 3
        assert stats["active_sources"] == 1

    def test_get_history(self):
        """Test getting conversation history as Q&A pairs"""
        mock_processor = Mock()
        session = ConversationSession(["test.txt"], processor=mock_processor)

        session.history.append(
            ConversationTurn(question="What is AI?", answer="Artificial Intelligence")
        )
        session.history.append(
            ConversationTurn(question="What is ML?", answer="Machine Learning")
        )

        history = session.get_history()

        assert history == [
            ("What is AI?", "Artificial Intelligence"),
            ("What is ML?", "Machine Learning"),
        ]

    def test_clear_history(self):
        """Test clearing conversation history"""
        mock_processor = Mock()
        session = ConversationSession(["test.txt"], processor=mock_processor)

        # Add some history
        session.history.append(
            ConversationTurn(question="What is AI?", answer="Artificial Intelligence")
        )

        assert len(session.history) == 1

        session.clear_history()

        assert len(session.history) == 0
        # Sources should be preserved
        assert session.sources == ["test.txt"]
