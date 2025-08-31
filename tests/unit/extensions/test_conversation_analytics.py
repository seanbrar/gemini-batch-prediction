from __future__ import annotations

import pytest

from gemini_batch.extensions.conversation import Conversation
from gemini_batch.extensions.conversation_types import ConversationState, Exchange

pytestmark = pytest.mark.unit


def test_conversation_analytics_summary() -> None:
    turns = (
        Exchange("Q1", "A1", error=False, estimate_max=100, actual_tokens=80),
        Exchange("Q2", "A2", error=True, estimate_max=50, actual_tokens=None),
    )
    conv = Conversation(ConversationState(sources=(), turns=turns), executor=None)  # type: ignore[arg-type]
    a = conv.analytics()
    assert a.total_turns == 2
    assert a.error_turns == 1
    assert a.total_estimated_tokens == 150
    assert a.total_actual_tokens == 80
