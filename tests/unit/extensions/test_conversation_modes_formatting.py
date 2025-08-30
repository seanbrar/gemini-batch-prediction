from __future__ import annotations

import pytest

from gemini_batch.extensions.conversation_modes import (
    SequentialMode,
    SingleMode,
    VectorizedMode,
)

pytestmark = pytest.mark.unit


def test_single_mode_normalizes_extras() -> None:
    mode = SingleMode()
    prompts = ("Q1", "Q2")
    answers = ("A1", "A2")
    exs = mode.format_exchanges(prompts, answers, {"warnings": ("w0",)})
    assert len(exs) == 1
    assert exs[0].user == "Q1"
    assert exs[0].assistant == "A1"
    # original warnings keep; normalization adds more warnings
    assert len(exs[0].warnings) >= 1


def test_sequential_mode_zips_and_warns() -> None:
    mode = SequentialMode()
    prompts = ("Q1", "Q2", "Q3")
    answers = ("A1",)
    exs = mode.format_exchanges(prompts, answers, {})
    assert len(exs) == 3
    assert exs[1].user == "Q2"
    assert exs[1].assistant == ""
    # warnings recorded on first exchange only
    assert len(exs[0].warnings) >= 1
    assert exs[1].warnings == ()


def test_vectorized_mode_combines() -> None:
    mode = VectorizedMode()
    prompts = ("Q1", "Q2")
    answers = ("A1", "A2")
    exs = mode.format_exchanges(prompts, answers, {})
    assert len(exs) == 1
    assert exs[0].user.startswith("[vectorized x2]")
    assert "A1" in exs[0].assistant and "A2" in exs[0].assistant
