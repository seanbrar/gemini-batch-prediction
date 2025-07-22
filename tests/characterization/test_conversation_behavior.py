"""
Characterization tests for the ConversationSession class.

These tests ensure that the behavior of a conversation, particularly how it
handles history and context, remains consistent across code changes.
"""

import json

import pytest

from gemini_batch import ConversationSession


@pytest.mark.golden_test("golden_files/test_conversation_basic.yml")
def test_conversation_ask_behavior(golden, mock_gemini_client):
    """
    Characterizes the behavior of a simple conversation flow.
    It tests the `ask` method and verifies the state of the session
    history after a few turns.
    """
    # Arrange
    sources = golden["input"]["sources"]
    questions = golden["input"]["questions"]

    # Configure the mock client to return different answers for each call
    # to make the conversation flow realistic.
    mock_responses = [
        {
            "text": json.dumps(["Answer to the first question."]),
            "usage_metadata": {
                "prompt_tokens": 50,
                "candidates_token_count": 10,
                "total_tokens": 60,
            },
        },
        {
            "text": json.dumps(
                ["Answer to the second question, considering the first."]
            ),
            "usage_metadata": {
                "prompt_tokens": 70,
                "candidates_token_count": 15,
                "total_tokens": 85,
            },
        },
    ]
    mock_gemini_client.generate_content.side_effect = mock_responses

    # We need to mock the processor used by the conversation session.
    # We can't inject a client directly, so we patch BatchProcessor.
    from gemini_batch import BatchProcessor

    mock_processor = BatchProcessor(_client=mock_gemini_client)

    # Act
    # Create the session with the mocked processor
    session = ConversationSession(sources, _processor=mock_processor)
    for q in questions:
        session.ask(q)

    # To characterize the result, we'll capture the final history.
    # We need to convert it to a serializable format (a list of dicts).
    # Timestamps are omitted for deterministic tests.
    final_history = [
        {
            "question": turn.question,
            "answer": turn.answer,
            "error": turn.error,
        }
        for turn in session.get_detailed_history()
    ]

    # Assert
    assert final_history == golden.out["final_history"]
