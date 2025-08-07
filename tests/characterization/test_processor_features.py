"""
Characterization tests for advanced features of the BatchProcessor.

These tests cover specific, critical behaviors like error handling,
comparison mode, and multi-source processing to ensure the library's
core features are stable and reliable.
"""

import json
from typing import Any

import pytest

from gemini_batch.exceptions import BatchProcessingError


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
def test_batch_processor_fallback_behavior(golden, mock_gemini_client, batch_processor):
    """
    Characterizes the behavior of the BatchProcessor when a batch API call
    fails and it must gracefully fall back to individual API calls.
    """
    # Arrange
    content = golden["input"]["content"]
    questions = golden["input"]["questions"]

    mock_gemini_client.generate_content.side_effect = [
        BatchProcessingError("Simulated API failure for batch call."),
        {
            "text": json.dumps(["Answer from individual call 1."]),
            "usage": {
                "prompt_tokens": 60,
                "candidates_token_count": 15,
                "total_tokens": 75,
            },
        },
        {
            "text": json.dumps(["Answer from individual call 2."]),
            "usage": {
                "prompt_tokens": 65,
                "candidates_token_count": 20,
                "total_tokens": 85,
            },
        },
    ]

    # Act
    # Use the batch_processor fixture which now returns the test adapter
    actual_output = batch_processor.process_questions(
        content, questions, return_usage=True
    )

    # Clean the output for deterministic comparison
    actual_output = _remove_timing_fields(actual_output)

    # Assert
    assert actual_output == golden.out["output"]


@pytest.mark.golden_test("golden_files/test_processor_comparison.yml")
def test_batch_processor_comparison_mode(golden, mock_gemini_client, batch_processor):
    """
    Characterizes the behavior of the BatchProcessor when `compare_methods=True`.
    This should result in both a batch call and individual calls being made.
    """
    # Arrange
    content = golden["input"]["content"]
    questions = golden["input"]["questions"]

    mock_gemini_client.generate_content.side_effect = [
        {
            "text": json.dumps(["Batch answer for Q1.", "Batch answer for Q2."]),
            "usage": {
                "prompt_tokens": 150,
                "candidates_token_count": 80,
                "total_tokens": 230,
            },
        },
        {
            "text": json.dumps(["Individual answer for Q1."]),
            "usage": {
                "prompt_tokens": 70,
                "candidates_token_count": 40,
                "total_tokens": 110,
            },
        },
        {
            "text": json.dumps(["Individual answer for Q2."]),
            "usage": {
                "prompt_tokens": 75,
                "candidates_token_count": 45,
                "total_tokens": 120,
            },
        },
    ]

    # Act
    # Use the batch_processor fixture which now returns the test adapter
    actual_output = batch_processor.process_questions(
        content,
        questions,
        compare_methods=True,
        return_usage=True,
    )

    # Clean the output for deterministic comparison
    actual_output = _remove_timing_fields(actual_output)

    # Assert
    assert actual_output == golden.out["output"]
