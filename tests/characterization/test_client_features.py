"""
Characterization test for explicit caching behavior in the new pipeline.
"""

import asyncio

import pytest

from gemini_batch.core.types import InitialCommand, Source


@pytest.mark.golden_test("golden_files/test_client_caching.yml")
def test_explicit_caching_behavior(golden, char_executor):
    """Verifies cache creation then reuse using the new executor.

    We record interactions via the test adapter injected into the pipeline.
    """
    # Arrange
    large_content = Source.from_text("A very large piece of text content..." * 1000)
    question = "What is the summary?"
    interaction_log: list[dict[str, object]] = []

    # Build executor with interaction logging enabled
    executor = char_executor.build(interaction_log=interaction_log)

    # Act: First call should create a cache and then generate
    cmd = InitialCommand(
        sources=(large_content,), prompts=(question,), config=executor.config
    )
    _ = asyncio.run(executor.execute(cmd))

    # Second call: should reuse the cache
    _ = asyncio.run(executor.execute(cmd))

    # Assert: sequence matches the golden interaction log
    assert interaction_log == golden.out["interaction_log"]
