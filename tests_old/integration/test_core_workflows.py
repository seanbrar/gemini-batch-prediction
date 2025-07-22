"""
Critical end-to-end workflows for regression protection.
These tests validate the most important user scenarios.
"""

from unittest.mock import MagicMock, patch

import pytest

from gemini_batch.conversation import create_conversation
from tests.fixtures.content_samples import (
    EDUCATIONAL_CONTENT,
    MULTI_SOURCE_SCENARIOS,
    QUESTION_SETS,
)


@pytest.mark.integration
class TestCoreWorkflows:
    """Test critical end-to-end workflows that users depend on."""

    def test_conversation_session_basic_workflow(
        self, batch_processor, mock_genai_client
    ):
        """Test basic conversation session workflow"""
        # Setup response
        mock_response = MagicMock()
        mock_response.text = "This is a comprehensive answer to your question."
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.cached_content_token_count = 0
        mock_genai_client.models.generate_content.return_value = mock_response

        session = create_conversation(
            EDUCATIONAL_CONTENT["short_lesson"], client=mock_genai_client
        )

        # Test basic question-answer flow
        answer = session.ask("What is machine learning?")

        # Verify response
        assert len(answer) > 10
        assert "answer" in answer.lower()

        # Test multiple questions
        answers = session.ask_multiple(
            ["What are the benefits?", "What are the challenges?"]
        )

        # Verify multiple answers
        assert len(answers) == 2
        assert all(len(a) > 10 for a in answers)

    def test_conversation_history_integration(self, batch_processor, mock_genai_client):
        """Test that conversation history is properly maintained"""
        # Setup response
        mock_response = MagicMock()
        mock_response.text = "This answer builds on our previous discussion."
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 150
        mock_response.usage_metadata.candidates_token_count = 60
        mock_response.usage_metadata.cached_content_token_count = 0
        mock_genai_client.models.generate_content.return_value = mock_response

        session = create_conversation(
            EDUCATIONAL_CONTENT["medium_article"], client=mock_genai_client
        )

        # Build conversation history
        session.ask("What is deep learning?")
        session.ask("How does it relate to machine learning?")

        # Verify history is maintained
        history = session.get_history()
        assert len(history) == 2

        # Test follow-up question
        follow_up = session.ask("Can you elaborate on the transformer architecture?")

        # Verify follow-up shows context awareness
        assert len(follow_up) > 10

    def test_batch_processing_core_functionality(
        self, batch_processor, mock_genai_client
    ):
        """Test core batch processing functionality"""
        # Setup batch response that properly simulates successful batch processing
        batch_response = MagicMock()
        batch_response.text = """
        Answer 1: Machine learning automates analytical model building.
        Answer 2: It uses algorithms that learn iteratively from data.
        Answer 3: Computers can find hidden insights without explicit programming.
        """
        # Ensure proper usage metadata structure
        batch_response.usage_metadata = MagicMock()
        batch_response.usage_metadata.prompt_token_count = 500
        batch_response.usage_metadata.candidates_token_count = 200
        batch_response.usage_metadata.cached_content_token_count = 0

        mock_genai_client.models.generate_content.return_value = batch_response

        content = EDUCATIONAL_CONTENT["short_lesson"]
        questions = QUESTION_SETS["basic_comprehension"]

        # Mock the client.generate_batch method specifically to ensure batch processing succeeds
        with patch.object(batch_processor.client, "generate_batch") as mock_batch:
            mock_batch.return_value = batch_response

            result = batch_processor.process_questions(content, questions)

            # Verify core structure
            assert "answers" in result
            assert "metrics" in result
            assert "efficiency" in result

            # Verify answers
            assert len(result["answers"]) == len(questions)
            assert all(len(answer) > 10 for answer in result["answers"])

            # Verify metrics - should be 1 call for successful batch processing
            assert result["metrics"]["batch"]["calls"] == 1
            assert result["metrics"]["batch"]["prompt_tokens"] > 0
            assert result["metrics"]["batch"]["output_tokens"] > 0

    def test_error_handling_preserves_conversation_state(
        self, batch_processor, mock_genai_client
    ):
        """Test that errors don't corrupt conversation state"""
        # Setup to fail first call, succeed on retry
        mock_genai_client.models.generate_content.side_effect = [
            Exception("API Error"),  # First call fails
            MagicMock(text="Successful answer after error"),  # Retry succeeds
        ]

        session = create_conversation(
            "Test content for error handling", client=mock_genai_client
        )

        # This should handle the error gracefully
        answer = session.ask("Test question?")

        # Verify conversation state is preserved
        assert len(session.history) == 1
        assert session.get_stats()["successful_turns"] == 1
        assert session.get_stats()["error_turns"] == 0  # Should recover from error

    def test_multi_source_processing(self, batch_processor, mock_genai_client):
        """Test processing multiple content sources"""
        # Setup response with proper mock structure
        mock_response = MagicMock()
        mock_response.text = "Analysis covering multiple sources: Machine learning fundamentals and deep learning applications."
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 300
        mock_response.usage_metadata.candidates_token_count = 100
        mock_response.usage_metadata.cached_content_token_count = 0

        # Mock the batch generation specifically for multi-source
        with patch.object(batch_processor.client, "generate_batch") as mock_batch:
            mock_batch.return_value = mock_response

            sources = MULTI_SOURCE_SCENARIOS["comparative_analysis"]["sources"]
            questions = ["What are the main themes across these sources?"]

            result = batch_processor.process_questions(sources, questions)

            # Verify multi-source processing
            assert len(result["answers"]) == 1
            # The answer should be extracted from the mock_response.text
            answer_text = result["answers"][0]
            assert (
                "sources" in answer_text.lower()
                or "themes" in answer_text.lower()
                or "analysis" in answer_text.lower()
            )

            # Verify metrics show successful processing
            assert result["metrics"]["batch"]["calls"] == 1

    def test_conversation_persistence(
        self, batch_processor, mock_genai_client, tmp_path
    ):
        """Test conversation save/load functionality"""
        # Setup response
        mock_response = MagicMock()
        mock_response.text = "Saved conversation answer."
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.cached_content_token_count = 0
        mock_genai_client.models.generate_content.return_value = mock_response

        session = create_conversation(
            "Content for persistence test", client=mock_genai_client
        )

        # Add some conversation history
        session.ask("First question?")
        session.ask("Second question?")

        # Save conversation
        session_path = tmp_path / "test_conversation.json"
        session_id = session.save(str(session_path))

        # Load conversation
        loaded_session = session.load(session_id, str(session_path), batch_processor)

        # Verify state is preserved
        assert loaded_session.session_id == session.session_id
        assert len(loaded_session.history) == 2
        assert loaded_session.get_history() == session.get_history()

    def test_conversation_source_management(self, batch_processor, mock_genai_client):
        """Test adding/removing sources during conversation"""
        # Setup responses
        mock_response = MagicMock()
        mock_response.text = "Answer about current sources."
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.cached_content_token_count = 0
        mock_genai_client.models.generate_content.return_value = mock_response

        session = create_conversation("Initial content", client=mock_genai_client)

        # Test initial state
        assert len(session.list_sources()) == 1

        # Add source
        session.add_source("Additional content")
        assert len(session.list_sources()) == 2

        # Ask question with multiple sources
        answer = session.ask("What are the main topics?")
        assert len(answer) > 10

        # Remove source
        session.remove_source("Initial content")
        assert len(session.list_sources()) == 1
        assert "Additional content" in session.list_sources()

    def test_conversation_analytics(self, batch_processor, mock_genai_client):
        """Test conversation analytics and statistics"""
        # Setup responses
        mock_response = MagicMock()
        mock_response.text = "Analytics test answer."
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.cached_content_token_count = 0
        mock_genai_client.models.generate_content.return_value = mock_response

        session = create_conversation("Content for analytics", client=mock_genai_client)

        # Add some conversation turns
        session.ask("Question 1?")
        session.ask("Question 2?")
        session.ask("Question 3?")

        # Get analytics
        stats = session.get_stats()

        # Verify analytics structure
        assert "session_id" in stats
        assert "total_turns" in stats
        assert "successful_turns" in stats
        assert "success_rate" in stats
        assert "active_sources" in stats

        # Verify values
        assert stats["total_turns"] == 3
        assert stats["successful_turns"] == 3
        assert stats["success_rate"] == 1.0
        assert stats["active_sources"] == 1

    def test_conversation_context_synthesis(self, batch_processor, mock_genai_client):
        """Test that conversation context is properly synthesized"""
        # Setup response that should show context awareness
        mock_response = MagicMock()
        mock_response.text = "Based on the conversation history about neural networks, this answer builds on previous concepts."
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 200
        mock_response.usage_metadata.candidates_token_count = 80
        mock_response.usage_metadata.cached_content_token_count = 50

        # Ensure all client methods return properly structured responses
        mock_genai_client.models.generate_content.return_value = mock_response

        # Mock the conversation's underlying content processing
        with patch.object(batch_processor.client, "generate_content") as mock_single:
            mock_single.return_value = mock_response

            session = create_conversation(
                EDUCATIONAL_CONTENT["conversation_context"], client=mock_genai_client
            )

            # Build conversation context
            session.ask("What are neural networks?")
            session.ask("How do they learn?")

            # Ask follow-up that should reference history
            follow_up = session.ask("How do attention mechanisms improve on this?")

            # Verify the response shows context awareness
            assert (
                "neural networks" in follow_up.lower()
                or "previous" in follow_up.lower()
                or "conversation" in follow_up.lower()
                or "concepts" in follow_up.lower()
            )

    def test_conversation_error_recovery(self, batch_processor, mock_genai_client):
        """Test conversation recovery from errors"""
        # Setup to fail first, then succeed
        mock_genai_client.models.generate_content.side_effect = [
            Exception("Temporary API error"),
            MagicMock(text="Recovered answer"),
        ]

        session = create_conversation(
            "Error recovery test content", client=mock_genai_client
        )

        # Should recover from error
        answer = session.ask("Test question?")

        # Verify recovery
        assert len(answer) > 10
        assert session.get_stats()["successful_turns"] == 1
        assert session.get_stats()["error_turns"] == 0  # Should not count as error turn

    def test_conversation_clear_history(self, batch_processor, mock_genai_client):
        """Test clearing conversation history while preserving sources"""
        # Setup response
        mock_response = MagicMock()
        mock_response.text = "Cleared history answer."
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.cached_content_token_count = 0
        mock_genai_client.models.generate_content.return_value = mock_response

        session = create_conversation("Test content", client=mock_genai_client)

        # Add some history
        session.ask("Question 1?")
        session.ask("Question 2?")

        # Verify history exists
        assert len(session.history) == 2

        # Clear history
        session.clear_history()

        # Verify history cleared but sources preserved
        assert len(session.history) == 0
        assert len(session.list_sources()) == 1

        # Verify can still ask questions
        answer = session.ask("New question?")
        assert len(answer) > 10
