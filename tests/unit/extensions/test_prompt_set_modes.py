from __future__ import annotations

import pytest

from gemini_batch.extensions.conversation_modes import (
    SequentialMode,
    SingleMode,
    VectorizedMode,
)
from gemini_batch.extensions.conversation_types import PromptSet

pytestmark = pytest.mark.unit


def test_prompt_set_constructors() -> None:
    single = PromptSet.single("Hello")
    assert single.mode == SingleMode()
    assert single.prompts == ("Hello",)

    seq = PromptSet.sequential("Q1", "Q2")
    assert seq.mode == SequentialMode()
    assert seq.prompts == ("Q1", "Q2")

    vec = PromptSet.vectorized("A", "B", "C")
    assert vec.mode == VectorizedMode()
    assert vec.prompts == ("A", "B", "C")
