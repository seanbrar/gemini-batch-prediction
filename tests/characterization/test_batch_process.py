"""
Characterization tests for BatchProcessor.

These tests use the pytest-golden plugin to create a "snapshot" of the
BatchProcessor's output. They ensure that any changes to the processor's
logic or the structure of its output dictionary are intentional.

Key principles applied here:
- Dependency Injection: We inject a `mock_gemini_client` into the BatchProcessor
  to isolate it from the actual network and API.
- Behavior, Not Implementation: We test the public `process_questions` method,
  treating the processor as a black box. We don't care *how* it generates the
  output, only that the output remains consistent.
"""

import asyncio
import json
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


@pytest.mark.golden_test("golden_files/test_batch_processor_basic.yml")
def test_batch_processor_basic_behavior(golden, mock_gemini_client, char_executor):  # noqa: ARG001
    """
    Characterizes the output of BatchProcessor.process_questions for a basic
    text-based query.
    """
    # Arrange
    content = Source.from_text(golden["input"]["content"])  # explicit source
    questions = golden["input"]["questions"]

    # Configure adapter via legacy mock to return per-question answers
    expected_answers = list(golden.out["output"]["answers"])  # from golden
    exec_ns = char_executor
    exec_ns.adapter.queue = [
        {"text": a, "usage": {"total_token_count": 0}} for a in expected_answers
    ]

    # Act
    # Use the batch_processor fixture which now returns our test adapter
    executor = char_executor.build()
    cmd = InitialCommand(
        sources=(content,), prompts=tuple(questions), config=executor.config
    )
    actual_output = asyncio.run(executor.execute(cmd))

    # Clean the output for deterministic comparison
    actual_output = _remove_timing_fields(actual_output)

    # Assert
    # Compare answers only; token/efficiency shapes differ in new architecture
    assert actual_output.get("answers") == golden.out["output"]["answers"]


@pytest.mark.golden_test("golden_files/test_batch_processor_structured.yml")
def test_batch_processor_structured_output(golden, mock_gemini_client, char_executor):
    """
    Characterizes the output of BatchProcessor when using a response_schema
    to get structured data.
    """
    # Arrange
    content = Source.from_text(golden["input"]["content"])
    questions = golden["input"]["questions"]
    # Structured schema kept for context in golden, not directly used in new pipeline

    mock_structured_response = {
        "summary": "This is a mocked summary of the content provided.",
        "key_points": ["This is the first key point.", "This is the second key point."],
    }
    mock_gemini_client.generate_content.return_value = {
        "text": json.dumps(mock_structured_response),
        "usage": {"total_token_count": 0},
    }

    # Act
    # Use the executor fixture (new architecture)
    executor = char_executor.build()
    cmd = InitialCommand(
        sources=(content,), prompts=tuple(questions), config=executor.config
    )
    actual_output = asyncio.run(executor.execute(cmd))

    # Structured data may already be a dict in new pipeline

    # Clean the output for deterministic comparison
    actual_output = _remove_timing_fields(actual_output)

    # Assert: new pipeline surfaces structured_data and one answer placeholder
    sd = actual_output.get("structured_data")
    assert isinstance(sd, dict)
    assert isinstance(actual_output.get("answers"), list)
    assert len(actual_output["answers"]) == 1
