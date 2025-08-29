from typing import Any

import pytest

from gemini_batch.config.core import FrozenConfig
from gemini_batch.core.exceptions import (
    GeminiBatchError,
    InvariantViolationError,
    PipelineError,
)
from gemini_batch.core.models import APITier
from gemini_batch.core.types import Failure, InitialCommand, Result, Success
from gemini_batch.executor import GeminiExecutor
from gemini_batch.pipeline.base import BaseAsyncHandler


class FailingStage(BaseAsyncHandler[Any, Any, GeminiBatchError]):
    """A minimal handler that always fails to exercise PipelineError path."""

    async def handle(
        self, _command: Any
    ) -> Result[Any, GeminiBatchError]:  # pragma: no cover - signature exercised by executor
        return Failure(GeminiBatchError("boom"))


class PassThroughStage(BaseAsyncHandler[Any, Any, GeminiBatchError]):
    """A minimal handler that returns the input unchanged, violating the final envelope invariant."""

    async def handle(
        self, command: Any
    ) -> Result[Any, GeminiBatchError]:  # pragma: no cover - signature exercised by executor
        return Success(command)


def _minimal_config() -> FrozenConfig:
    return FrozenConfig(
        model="gemini-2.0-flash",
        api_key=None,
        use_real_api=False,
        enable_caching=False,
        ttl_seconds=0,
        telemetry_enabled=False,
        tier=APITier.FREE,
        provider="gemini",
        extra={},
    )


def _minimal_command(cfg: FrozenConfig | None = None) -> InitialCommand:
    return InitialCommand(
        sources=(), prompts=("Hello",), config=cfg or _minimal_config()
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pipeline_error_uses_true_stage_name() -> None:
    executor = GeminiExecutor(_minimal_config(), pipeline_handlers=[FailingStage()])

    with pytest.raises(PipelineError) as ei:
        await executor.execute(_minimal_command())

    err = ei.value
    # Stage name should reflect the inner stage, not the erased wrapper
    assert err.handler_name == "FailingStage"
    # Convenience check: stage_names property exposes correct names
    assert executor.stage_names == ("FailingStage",)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invariant_violation_when_no_result_envelope_produced() -> None:
    # A single pass-through stage leaves the executor with a non-dict state
    executor = GeminiExecutor(_minimal_config(), pipeline_handlers=[PassThroughStage()])

    with pytest.raises(InvariantViolationError) as ei:
        await executor.execute(_minimal_command())

    assert "ResultEnvelope" in str(ei.value)
    # Stage name recorded on the invariant error should be the final stage
    assert getattr(ei.value, "stage_name", None) == "PassThroughStage"


@pytest.mark.unit
def test_erase_guard_raises_for_invalid_handler() -> None:
    # Creating an executor with an invalid handler (no 'handle') should raise TypeError during erase()
    with pytest.raises(TypeError):
        GeminiExecutor(_minimal_config(), pipeline_handlers=[object()])  # type: ignore
