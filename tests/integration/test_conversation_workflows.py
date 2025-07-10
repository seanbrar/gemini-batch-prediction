"""
Integration tests for conversation workflows.
"""

import json
import tempfile
from unittest.mock import Mock, patch

import pytest

from gemini_batch.conversation import create_conversation
from tests.fixtures.content_samples import EDUCATIONAL_CONTENT


@pytest.mark.integration
class TestConversationWorkflows:
    """Test conversation workflow scenarios"""

    def test_conversation_history_is_passed_as_context(
        self, batch_processor, mock_genai_client
    ):
        """Test that conversation history is passed as context to subsequent questions."""
        # Setup responses for different turns
        first_response = Mock()
        first_response.text = json.dumps(["Machine learning is a subset of AI."])
        first_response.usage_metadata = Mock()
        first_response.usage_metadata.prompt_token_count = 100
        first_response.usage_metadata.candidates_token_count = 50
        first_response.usage_metadata.cached_content_token_count = 0

        second_response = Mock()
        second_response.text = json.dumps(
            ["Deep learning builds on machine learning concepts."]
        )
        second_response.usage_metadata = Mock()
        second_response.usage_metadata.prompt_token_count = 150
        second_response.usage_metadata.candidates_token_count = 75
        second_response.usage_metadata.cached_content_token_count = 0

        # Setup mock to return different responses
        mock_genai_client.generate_content.side_effect = [
            first_response,
            second_response,
        ]

        session = create_conversation(
            EDUCATIONAL_CONTENT["short_lesson"],
            client=mock_genai_client,
        )

        # First question
        first_answer = session.ask("What is machine learning?")

        # Second question that should reference previous context
        second_answer = session.ask("How does deep learning relate to this?")

        # Verify both questions were answered
        assert len(first_answer) > 0
        assert len(second_answer) > 0

        # Verify history is maintained
        history = session.get_history()
        assert len(history) == 2

        # Verify the second call included context from the first
        assert mock_genai_client.generate_content.call_count == 2

        # Check that the second call included more content (due to history)
        second_call_args = mock_genai_client.generate_content.call_args_list[1]
        # The second call should have more content due to history inclusion
        assert "machine learning" in str(second_call_args).lower()

    def test_add_source_mid_conversation(self, batch_processor, mock_genai_client):
        """Test adding a source mid-conversation."""
        # Setup response
        response = Mock()
        response.text = json.dumps(["Answer using both sources"])
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 200
        response.usage_metadata.candidates_token_count = 100
        response.usage_metadata.cached_content_token_count = 0

        mock_genai_client.generate_content.return_value = response

        session = create_conversation(
            EDUCATIONAL_CONTENT["short_lesson"],
            client=mock_genai_client,
        )

        # Add a second source
        session.add_source(EDUCATIONAL_CONTENT["medium_article"])

        # Ask a question that should use both sources
        answer = session.ask("What are the key differences between these approaches?")

        # Verify answer was generated
        assert len(answer) > 0

        # Verify the call included both sources
        call_args = mock_genai_client.generate_content.call_args
        call_content = str(call_args)

        # Should include content from both sources
        assert "machine learning" in call_content.lower()
        assert "deep learning" in call_content.lower()

    def test_conversation_save_and_load(self, batch_processor, mock_genai_client):
        """Test conversation save and load functionality."""
        # Setup response
        response = Mock()
        response.text = json.dumps(["Saved conversation answer"])
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 50
        response.usage_metadata.cached_content_token_count = 0

        mock_genai_client.generate_content.return_value = response

        session = create_conversation(
            EDUCATIONAL_CONTENT["short_lesson"],
            client=mock_genai_client,
        )

        # Add some conversation history
        session.ask("What is AI?")
        session.ask("How does it work?")

        # Save conversation
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            session.save(f.name)
            save_path = f.name

        # Create new session and load
        new_session = create_conversation(client=batch_processor)
        new_session.load(save_path)

        # Verify history was loaded
        history = new_session.get_history()
        assert len(history) == 2

        # Verify sources were loaded
        sources = new_session.get_sources()
        assert len(sources) > 0

        # Clean up
        import os

        os.unlink(save_path)

    def test_conversation_clear_history(self, batch_processor, mock_genai_client):
        """Test clearing conversation history."""
        # Setup response
        response = Mock()
        response.text = json.dumps(["Answer after clearing history"])
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 50
        response.usage_metadata.cached_content_token_count = 0

        mock_genai_client.generate_content.return_value = response

        session = create_conversation(
            EDUCATIONAL_CONTENT["short_lesson"],
            client=mock_genai_client,
        )

        # Add some history
        session.ask("What is AI?")
        session.ask("How does it work?")

        # Verify history exists
        assert len(session.get_history()) == 2

        # Clear history
        session.clear_history()

        # Verify history is cleared
        assert len(session.get_history()) == 0

        # Verify sources are still available
        sources = session.get_sources()
        assert len(sources) > 0

    def test_conversation_statistics_tracking(self, batch_processor, mock_genai_client):
        """Test conversation statistics tracking."""
        # Setup responses
        responses = [
            Mock(
                text=json.dumps([f"Answer {i}"]),
                usage_metadata=Mock(
                    prompt_token_count=100,
                    candidates_token_count=50,
                    cached_content_token_count=0,
                ),
            )
            for i in range(3)
        ]

        mock_genai_client.generate_content.side_effect = responses

        session = create_conversation(
            EDUCATIONAL_CONTENT["short_lesson"],
            client=mock_genai_client,
        )

        # Ask multiple questions
        session.ask("What is AI?")
        session.ask("How does it work?")
        session.ask("What are the applications?")

        # Get statistics
        stats = session.get_stats()

        # Verify statistics
        assert stats["total_turns"] == 3
        assert stats["successful_turns"] == 3
        assert stats["error_turns"] == 0
        assert stats["total_tokens"] > 0
        assert stats["average_tokens_per_turn"] > 0

    def test_conversation_error_recovery(self, batch_processor, mock_genai_client):
        """Test conversation error recovery."""
        # Setup first call to fail, second to succeed
        mock_genai_client.generate_content.side_effect = [
            Exception("API Error"),
            Mock(
                text=json.dumps(["Recovered answer"]),
                usage_metadata=Mock(
                    prompt_token_count=100,
                    candidates_token_count=50,
                    cached_content_token_count=0,
                ),
            ),
        ]

        session = create_conversation(
            EDUCATIONAL_CONTENT["short_lesson"],
            client=mock_genai_client,
        )

        # First question should fail
        with pytest.raises(Exception):
            session.ask("What is AI?")

        # Second question should succeed
        answer = session.ask("What is machine learning?")

        # Verify recovery
        assert len(answer) > 0

        # Verify statistics show error
        stats = session.get_stats()
        assert stats["error_turns"] == 1
        assert stats["successful_turns"] == 1

    def test_conversation_context_synthesis(self, batch_processor, mock_genai_client):
        """Test conversation context synthesis."""
        # Setup response that shows context awareness
        response = Mock()
        response.text = json.dumps(
            [
                "Based on our previous discussion about AI and machine learning, deep learning is the next evolution."
            ]
        )
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 200
        response.usage_metadata.candidates_token_count = 100
        response.usage_metadata.cached_content_token_count = 0

        mock_genai_client.generate_content.return_value = response

        session = create_conversation(
            EDUCATIONAL_CONTENT["short_lesson"],
            client=mock_genai_client,
        )

        # Build context through conversation
        session.ask("What is artificial intelligence?")
        session.ask("How does machine learning fit into AI?")

        # Ask a question that should synthesize previous context
        answer = session.ask("What comes after machine learning?")

        # Verify context synthesis
        assert len(answer) > 0
        assert "previous discussion" in answer.lower() or "based on" in answer.lower()

    def test_conversation_multiple_sources(self, batch_processor, mock_genai_client):
        """Test conversation with multiple sources."""
        # Setup response
        response = Mock()
        response.text = json.dumps(["Analysis covering multiple sources"])
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 300
        response.usage_metadata.candidates_token_count = 150
        response.usage_metadata.cached_content_token_count = 0

        mock_genai_client.generate_content.return_value = response

        session = create_conversation(
            [
                EDUCATIONAL_CONTENT["short_lesson"],
                EDUCATIONAL_CONTENT["medium_article"],
            ],
            client=mock_genai_client,
        )

        # Ask question about multiple sources
        answer = session.ask("What are the key differences between these approaches?")

        # Verify answer was generated
        assert len(answer) > 0

        # Verify both sources were used
        call_args = mock_genai_client.generate_content.call_args
        call_content = str(call_args)

        # Should include content from both sources
        assert "machine learning" in call_content.lower()
        assert "deep learning" in call_content.lower()

    def test_conversation_persistence_with_metadata(
        self, batch_processor, mock_genai_client
    ):
        """Test conversation persistence with metadata."""
        # Setup response
        response = Mock()
        response.text = json.dumps(["Answer with metadata"])
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 50
        response.usage_metadata.cached_content_token_count = 0

        mock_genai_client.generate_content.return_value = response

        session = create_conversation(
            EDUCATIONAL_CONTENT["short_lesson"],
            client=mock_genai_client,
        )

        # Add metadata
        session.metadata["session_id"] = "test_session_123"
        session.metadata["user_id"] = "test_user"

        # Add conversation history
        session.ask("What is AI?")

        # Save with metadata
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            session.save(f.name)
            save_path = f.name

        # Load and verify metadata
        new_session = create_conversation(client=batch_processor)
        new_session.load(save_path)

        # Verify metadata was preserved
        assert new_session.metadata["session_id"] == "test_session_123"
        assert new_session.metadata["user_id"] == "test_user"

        # Clean up
        import os

        os.unlink(save_path)

    def test_conversation_rate_limiting(self, batch_processor, mock_genai_client):
        """Test conversation rate limiting."""
        # Setup rate limiter
        with patch.object(batch_processor.client, "rate_limiter") as mock_rate_limiter:
            mock_rate_limiter.wait_if_needed.return_value = None

            # Setup response
            response = Mock()
            response.text = json.dumps(["Rate limited answer"])
            response.usage_metadata = Mock()
            response.usage_metadata.prompt_token_count = 100
            response.usage_metadata.candidates_token_count = 50
            response.usage_metadata.cached_content_token_count = 0

            mock_genai_client.generate_content.return_value = response

            session = create_conversation(
                EDUCATIONAL_CONTENT["short_lesson"],
                client=mock_genai_client,
            )

            # Ask multiple questions rapidly
            session.ask("What is AI?")
            session.ask("How does it work?")
            session.ask("What are the applications?")

            # Verify rate limiting was called
            assert mock_rate_limiter.wait_if_needed.call_count == 3

            # Verify all questions were answered
            assert len(session.get_history()) == 3

    def test_conversation_source_management(self, batch_processor, mock_genai_client):
        """Test conversation source management."""
        # Setup response
        response = Mock()
        response.text = json.dumps(["Answer with managed sources"])
        response.usage_metadata = Mock()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 50
        response.usage_metadata.cached_content_token_count = 0

        mock_genai_client.generate_content.return_value = response

        session = create_conversation(
            EDUCATIONAL_CONTENT["short_lesson"],
            client=mock_genai_client,
        )

        # Verify initial sources
        initial_sources = session.list_sources()
        assert len(initial_sources) > 0

        # Add a source
        session.add_source(EDUCATIONAL_CONTENT["medium_article"])

        # Verify source was added
        updated_sources = session.list_sources()
        assert len(updated_sources) > len(initial_sources)

        # Remove a source
        session.remove_source(0)  # Remove first source

        # Verify source was removed
        final_sources = session.list_sources()
        assert len(final_sources) == 1

    def test_conversation_analytics(self, batch_processor, mock_genai_client):
        """Test conversation analytics and insights."""
        # Setup responses with different characteristics
        responses = [
            Mock(
                text=json.dumps(["Short answer"]),
                usage_metadata=Mock(
                    prompt_token_count=50,
                    candidates_token_count=25,
                    cached_content_token_count=0,
                ),
            ),
            Mock(
                text=json.dumps(
                    ["Long detailed answer about machine learning concepts"]
                ),
                usage_metadata=Mock(
                    prompt_token_count=200,
                    candidates_token_count=100,
                    cached_content_token_count=0,
                ),
            ),
            Mock(
                text=json.dumps(["Medium answer"]),
                usage_metadata=Mock(
                    prompt_token_count=100,
                    candidates_token_count=50,
                    cached_content_token_count=0,
                ),
            ),
        ]

        mock_genai_client.generate_content.side_effect = responses

        session = create_conversation(
            EDUCATIONAL_CONTENT["short_lesson"],
            client=mock_genai_client,
        )

        # Ask questions with different characteristics
        session.ask("What is AI?")
        session.ask("Explain machine learning in detail")
        session.ask("What are neural networks?")

        # Get analytics
        analytics = session.get_analytics()

        # Verify analytics
        assert "total_turns" in analytics
        assert "average_response_length" in analytics
        assert "total_tokens" in analytics
        assert "token_efficiency" in analytics

        # Verify calculations
        assert analytics["total_turns"] == 3
        assert analytics["total_tokens"] > 0
        assert analytics["average_response_length"] > 0
