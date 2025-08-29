"""Demonstrate dev-time validation with GEMINI_PIPELINE_VALIDATE=1.

Run:

- GEMINI_PIPELINE_VALIDATE=1 python examples/validation_trigger_demo.py

This script executes the pipeline twice:
1) A normal, healthy pipeline to show a successful result.
2) A pipeline with a tiny custom handler that corrupts telemetry in-flight,
   which triggers the dev-time envelope validator inside ResultBuilder.

The second run is intentionally caught and reported as an expected validation
success, keeping the example green while showcasing the validation behavior.
"""

from __future__ import annotations

import asyncio
from typing import Any

from gemini_batch._dev_flags import dev_validate_enabled
from gemini_batch.config import FrozenConfig, resolve_config
from gemini_batch.core.exceptions import InvariantViolationError
from gemini_batch.core.types import (
    InitialCommand,
    Source,
)
from gemini_batch.executor import GeminiExecutor
from gemini_batch.pipeline._devtools import validate_result_envelope
from gemini_batch.pipeline.api_handler import APIHandler
from gemini_batch.pipeline.cache_stage import CacheStage
from gemini_batch.pipeline.planner import ExecutionPlanner
from gemini_batch.pipeline.rate_limit_handler import RateLimitHandler
from gemini_batch.pipeline.registries import CacheRegistry, FileRegistry
from gemini_batch.pipeline.result_builder import ResultBuilder
from gemini_batch.pipeline.source_handler import SourceHandler


def _build_pipeline(_config: FrozenConfig) -> list[Any]:
    """Construct a minimal pipeline with optional telemetry corruption.

    Args:
        config: The frozen configuration for the run.

    Returns:
        The handler list to pass to the executor.
    """
    cache_registry = CacheRegistry()
    file_registry = FileRegistry()

    handlers: list[Any] = [
        SourceHandler(),
        ExecutionPlanner(),
        RateLimitHandler(),
        CacheStage(
            registries={"cache": cache_registry},
            adapter_factory=None,
        ),
        APIHandler(
            telemetry=None,
            registries={"cache": cache_registry, "files": file_registry},
            adapter_factory=None,
        ),
    ]

    # ResultBuilder picks up GEMINI_PIPELINE_VALIDATE=1 automatically.
    handlers.append(ResultBuilder(validate=None))
    return handlers


async def _run_pipeline_ok(config: FrozenConfig) -> None:
    """Run the executor and print a concise outcome line.

    Args:
        config: Frozen configuration for the executor.
        None
    """
    executor = GeminiExecutor(
        config,
        pipeline_handlers=_build_pipeline(config),
    )

    initial = InitialCommand.strict(
        sources=(Source.from_text("Example content for validation demo."),),
        prompts=("What is 2 + 2?",),
        config=config,
    )

    try:
        result = await executor.execute(initial)
        # Minimal success signal; demonstrates a complete envelope.
        answers = ", ".join(result.get("answers", []))
        print(
            f"OK: success={result.get('success')} answers=[{answers}] method={result.get('extraction_method')}"
        )
    except InvariantViolationError as e:
        # With the healthy pipeline, this should not occur even with validation on.
        print(f"UNEXPECTED VALIDATION ERROR: {e}")


def _demo_direct_validator() -> None:
    """Trigger the dev-time validator with a crafted bad envelope.

    This demonstrates how the validation raises under GEMINI_PIPELINE_VALIDATE=1
    with a precise reason, without relying on upstream pipeline internals.
    """
    bad_envelope: dict[str, Any] = {
        "success": True,
        "answers": ["ok"],
        "extraction_method": "demo",
        "confidence": 0.9,
        # Invalid: durations values must be int|float, not str.
        "metrics": {"durations": {"DemoStage": "not-a-number"}},
    }
    try:
        validate_result_envelope(bad_envelope, stage_name="Demo")
        print("UNEXPECTED: validator accepted bad envelope")
    except InvariantViolationError as e:
        print(f"VALIDATION TRIGGERED: {e}")


def main() -> None:
    """Entry point for the validation demo."""
    config = resolve_config()
    # 1) Healthy pipeline: prints an OK line.
    asyncio.run(_run_pipeline_ok(config))
    # 2) Dev validator: explicitly trigger a validation error when enabled.
    if dev_validate_enabled():
        _demo_direct_validator()
    else:
        print("Set GEMINI_PIPELINE_VALIDATE=1 to run the validator demo.")


if __name__ == "__main__":
    main()
