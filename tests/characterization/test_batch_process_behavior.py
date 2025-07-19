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

import json

import pytest

from gemini_batch import BatchProcessor
from tests.conftest import SimpleSummary


def _remove_timing_fields(output_dict):  # noqa: ANN001, ANN202
    """Remove timing-sensitive fields from output for consistent golden test comparisons."""  # noqa: E501
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
def test_batch_processor_basic_behavior(golden, mock_gemini_client):  # noqa: ANN001, ANN201
    """
    Characterizes the output of BatchProcessor.process_questions for a basic
    text-based query.
    """
    # Arrange
    content = golden["input"]["content"]
    questions = golden["input"]["questions"]

    mock_response_text = json.dumps(
        ["Mocked answer for question 1", "Mocked answer for question 2"]  # noqa: COM812
    )
    mock_gemini_client.generate_content.return_value = {
        "text": mock_response_text,
        "usage_metadata": {
            "prompt_tokens": 110,
            "candidates_token_count": 45,
            "total_tokens": 155,
        },
    }

    # Act
    processor = BatchProcessor(_client=mock_gemini_client)
    actual_output = processor.process_questions(content, questions, return_usage=True)

    # Clean the output for deterministic comparison
    actual_output = _remove_timing_fields(actual_output)

    # Assert
    assert actual_output == golden.out["output"]


@pytest.mark.golden_test("golden_files/test_batch_processor_structured.yml")
def test_batch_processor_structured_output(golden, mock_gemini_client):  # noqa: ANN001, ANN201
    """
    Characterizes the output of BatchProcessor when using a response_schema
    to get structured data.
    """
    # Arrange
    content = golden["input"]["content"]
    questions = golden["input"]["questions"]
    response_schema = SimpleSummary

    mock_structured_response = {
        "summary": "This is a mocked summary of the content provided.",
        "key_points": ["This is the first key point.", "This is the second key point."],
    }
    mock_gemini_client.generate_content.return_value = {
        "text": json.dumps(mock_structured_response),
        "usage_metadata": {
            "prompt_tokens": 120,
            "candidates_token_count": 60,
            "total_tokens": 180,
        },
    }

    # Act
    processor = BatchProcessor(_client=mock_gemini_client)
    actual_output = processor.process_questions(
        content,
        questions,
        response_schema=response_schema,
        return_usage=True,
    )

    # Convert 'structured_data' from Pydantic model to dict for global files YAML.
    if "structured_data" in actual_output and actual_output["structured_data"]:  # noqa: RUF019
        actual_output["structured_data"] = actual_output["structured_data"].model_dump()

    # Clean the output for deterministic comparison
    actual_output = _remove_timing_fields(actual_output)

    # Assert
    assert actual_output == golden.out["output"]
