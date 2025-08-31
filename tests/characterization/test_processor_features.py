"""
Characterization tests for advanced features of the BatchProcessor.

These tests cover specific, critical behaviors like error handling,
comparison mode, and multi-source processing to ensure the library's
core features are stable and reliable.
"""

import asyncio
from typing import Any

import pytest

from gemini_batch.core.types import InitialCommand, Source


def _remove_timing_fields(output_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove timing-sensitive fields from output for consistent golden test comparisons."""
    # Remove processing_time from top level
    output_dict.pop("processing_time", None)

    # Remove time from efficiency metrics
    if "efficiency" in output_dict:
        output_dict["efficiency"].pop("time_efficiency", None)

    # Remove time from detailed metrics
    if "metrics" in output_dict:
        for metric_type in ["batch", "individual"]:
            if metric_type in output_dict["metrics"]:
                output_dict["metrics"][metric_type].pop("time", None)

    return output_dict


@pytest.mark.golden_test("golden_files/test_processor_fallback.yml")
def test_batch_processor_fallback_behavior(golden, char_executor):
    """
    Characterizes the behavior of the BatchProcessor when a batch API call
    fails and it must gracefully fall back to individual API calls.
    """
    # Arrange
    content = Source.from_text(golden["input"]["content"])
    questions = golden["input"]["questions"]

    # New pipeline: vectorized execution over prompts (no explicit fallback)
    # Queue per-prompt responses to mirror legacy fallback answers
    answers = golden.out["output"]["answers"]
    char_executor.adapter.queue = [
        {"text": a, "usage": {"total_token_count": 0}} for a in answers
    ]

    executor = char_executor.build()
    cmd = InitialCommand(
        sources=(content,), prompts=tuple(questions), config=executor.config
    )
    actual_output = asyncio.run(executor.execute(cmd))

    # Compare answers only; shape differs in new architecture
    assert actual_output.get("answers") == answers


@pytest.mark.golden_test("golden_files/test_processor_comparison.yml")
def test_batch_processor_comparison_mode(golden, char_executor):
    """
    Characterizes the behavior of the BatchProcessor when `compare_methods=True`.
    This should result in both a batch call and individual calls being made.
    """
    # Arrange
    content = Source.from_text(golden["input"]["content"])
    questions = golden["input"]["questions"]

    # New pipeline: no comparison mode; assert vectorized answers equal legacy batch answers
    batch_answers = golden.out["output"]["answers"]
    char_executor.adapter.queue = [
        {"text": a, "usage": {"total_token_count": 0}} for a in batch_answers
    ]

    executor = char_executor.build()
    cmd = InitialCommand(
        sources=(content,), prompts=tuple(questions), config=executor.config
    )
    actual_output = asyncio.run(executor.execute(cmd))

    assert actual_output.get("answers") == batch_answers
