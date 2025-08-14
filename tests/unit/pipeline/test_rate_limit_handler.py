from unittest.mock import Mock, patch

import pytest

from gemini_batch.core.types import (
    APICall,
    ExecutionPlan,
    PlannedCommand,
    RateConstraint,
    ResolvedCommand,
    Success,
    TextPart,
)
from gemini_batch.pipeline.rate_limit_handler import RateLimitHandler


def _planned_with_constraint(rpm: int, tpm: int | None = None) -> PlannedCommand:
    initial = Mock()
    initial.config = {"model": "gemini-2.0-flash"}
    resolved = ResolvedCommand(initial=initial, resolved_sources=())
    call = APICall(
        model_name="gemini-2.0-flash", api_parts=(TextPart("hi"),), api_config={}
    )
    plan = ExecutionPlan(primary_call=call, rate_constraint=RateConstraint(rpm, tpm))
    return PlannedCommand(resolved=resolved, execution_plan=plan)


@pytest.mark.asyncio
async def test_passthrough_without_constraint():
    handler = RateLimitHandler()
    initial = Mock()
    initial.config = {"model": "gemini-2.0-flash"}
    resolved = ResolvedCommand(initial=initial, resolved_sources=())
    call = APICall(
        model_name="gemini-2.0-flash", api_parts=(TextPart("hi"),), api_config={}
    )
    plan = ExecutionPlan(primary_call=call, rate_constraint=None)
    cmd = PlannedCommand(resolved=resolved, execution_plan=plan)

    result = await handler.handle(cmd)
    assert isinstance(result, Success)
    assert result.value is cmd


@pytest.mark.asyncio
async def test_request_delay_enforced_with_mocked_clock():
    # Simulate second request at time 0 with rpm=60 → expect 1s sleep
    times = [0.0, 0.0, 0.0]
    clock = Mock(side_effect=times)
    handler = RateLimitHandler(clock=clock)
    cmd = _planned_with_constraint(rpm=60)

    with patch("asyncio.sleep") as sleep_mock:
        # First pass: no wait (initial last_time=0)
        await handler.handle(cmd)
        # Second pass: should wait ~1.0s
        await handler.handle(cmd)
        assert sleep_mock.called


@pytest.mark.asyncio
async def test_token_delay_enforced_with_mocked_clock():
    # For tpm=120 and estimated tokens=60, required ~30s → ensure a sleep occurs
    times = [0.0, 0.0, 0.0]
    clock = Mock(side_effect=times)
    handler = RateLimitHandler(clock=clock)
    cmd = _planned_with_constraint(rpm=1000, tpm=120)
    object.__setattr__(cmd, "token_estimate", Mock(max_tokens=60))

    with patch("asyncio.sleep") as sleep_mock:
        await handler.handle(cmd)
        assert sleep_mock.called


@pytest.mark.asyncio
async def test_key_extractor_normalizes_tier_enum_or_string():
    handler = RateLimitHandler()

    # Case 1: tier as enum-like object with .value
    enum_like = Mock()
    enum_like.value = "free"
    initial1 = Mock()
    initial1.config = {"model": "gemini-2.0-flash", "tier": enum_like}
    resolved1 = ResolvedCommand(initial=initial1, resolved_sources=())
    call1 = APICall(
        model_name="gemini-2.0-flash", api_parts=(TextPart("hi"),), api_config={}
    )
    plan1 = ExecutionPlan(primary_call=call1)
    cmd1 = PlannedCommand(resolved=resolved1, execution_plan=plan1)
    key1 = handler._default_key_extractor(cmd1)
    assert key1[2] == "free"

    # Case 2: tier as plain string
    initial2 = Mock()
    initial2.config = {"model": "gemini-2.0-flash", "tier": "tier_1"}
    resolved2 = ResolvedCommand(initial=initial2, resolved_sources=())
    call2 = APICall(
        model_name="gemini-2.0-flash", api_parts=(TextPart("hi"),), api_config={}
    )
    plan2 = ExecutionPlan(primary_call=call2)
    cmd2 = PlannedCommand(resolved=resolved2, execution_plan=plan2)
    key2 = handler._default_key_extractor(cmd2)
    assert key2[2] == "tier_1"
