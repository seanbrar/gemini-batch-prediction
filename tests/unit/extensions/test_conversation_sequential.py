"""Tests for sequential mode in the Conversation extension.

Verifies that sequential execution appends one Exchange per prompt and
produces per-prompt metrics and aggregated totals.
"""

from __future__ import annotations

import pytest

from gemini_batch.executor import create_executor
from gemini_batch.extensions.conversation import Conversation
from gemini_batch.extensions.conversation_types import PromptSet

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_conversation_sequential_appends_per_prompt_turns() -> None:
    executor = create_executor()
    conv = Conversation.start(executor)

    prompts = ("First question?", "Second question?")
    conv2, answers, metrics = await conv.run(PromptSet(prompts, "sequential"))

    # Two new turns appended
    assert len(conv2.state.turns) == 2
    assert conv2.state.turns[0].user == prompts[0]
    assert conv2.state.turns[1].user == prompts[1]

    # Answers align to prompts
    assert len(answers) == 2

    # Per-prompt metrics present with two entries; totals contain numeric aggregates
    assert hasattr(metrics, "per_prompt") and len(metrics.per_prompt) == 2
    assert hasattr(metrics, "totals") and isinstance(metrics.totals, dict)

