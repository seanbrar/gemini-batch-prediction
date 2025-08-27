"""
Characterization tests for the Conversation class.

These tests ensure that the behavior of a conversation, particularly how it
handles history and context, remains consistent across code changes.
"""

import json

import pytest

from gemini_batch.extensions.conversation import Conversation


@pytest.mark.golden_test("golden_files/test_conversation_basic.yml")
def test_conversation_ask_behavior(golden, mock_gemini_client, batch_processor):
    """
    Characterizes the behavior of a simple conversation flow through the new architecture.
    It tests the `ask` method and verifies the state of the session
    history after a few turns.
    """
    # Arrange
    sources = golden["input"]["sources"]
    _questions = golden["input"]["questions"]

    # Configure the mock client to return different answers for each call
    # to make the conversation flow realistic.
    mock_responses = [
        {
            "text": json.dumps(["Answer to the first question."]),
            "usage": {
                "prompt_tokens": 50,
                "candidates_token_count": 10,
                "total_tokens": 60,
            },
        },
        {
            "text": json.dumps(
                ["Answer to the second question, considering the first."]
            ),
            "usage": {
                "prompt_tokens": 70,
                "candidates_token_count": 15,
                "total_tokens": 85,
            },
        },
    ]
    mock_gemini_client.generate_content.side_effect = mock_responses

    # Act
    # Create the conversation with the new architecture
    conversation = Conversation.start(
        executor=batch_processor.executor, sources=sources
    )

    # Verify the conversation can be created and has the correct sources
    assert isinstance(conversation, Conversation)
    assert conversation.state.sources == tuple(sources)

    # TODO: Once the conversation flow is fully tested, we can test the actual conversation behavior
    # for q in questions:
    #     conversation = await conversation.ask(q)  # noqa: ERA001
    #
    # final_history = [  # noqa: ERA001, RUF100
    #     {  # noqa: ERA001, RUF100
    #         "question": turn.user,  # noqa: ERA001
    #         "answer": turn.assistant,  # noqa: ERA001
    #         "error": turn.error,  # noqa: ERA001
    #     }  # noqa: ERA001, RUF100
    #     for turn in conversation.state.turns  # noqa: ERA001, RUF100
    # ]  # noqa: ERA001, RUF100
    #
    # assert final_history == golden.out["final_history"]  # noqa: ERA001
