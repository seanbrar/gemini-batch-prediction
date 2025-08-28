import asyncio

import pytest

from gemini_batch.config.core import FrozenConfig
from gemini_batch.core.models import APITier
from gemini_batch.core.types import (
    InitialCommand,
    PlannedCommand,
    ResolvedCommand,
    Source,
)
from gemini_batch.pipeline.planner import ExecutionPlanner

pytestmark = pytest.mark.unit


def _cfg() -> FrozenConfig:
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


async def _plan_with_sources(*sources: Source) -> PlannedCommand:
    init = InitialCommand.strict(
        sources=tuple(sources),
        prompts=("Hello",),
        config=_cfg(),
    )
    resolved = ResolvedCommand(initial=init, resolved_sources=tuple(sources))
    planner = ExecutionPlanner()
    result = await planner.handle(resolved)
    assert hasattr(result, "value"), getattr(result, "error", None)
    return result.value


def test_planner_maps_youtube_to_filerefp():
    s = Source.from_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    planned = asyncio.get_event_loop().run_until_complete(_plan_with_sources(s))
    # shared_parts should contain a FileRefPart with file_uri
    shared = planned.execution_plan.shared_parts
    assert any(
        getattr(p, "uri", "").startswith("http")
        and getattr(p, "mime_type", None) == "video/youtube"
        for p in shared
    )


def test_planner_maps_arxiv_to_filerefp():
    s = Source.from_arxiv("1706.03762v3")
    planned = asyncio.get_event_loop().run_until_complete(_plan_with_sources(s))
    shared = planned.execution_plan.shared_parts
    assert any(
        getattr(p, "uri", "").endswith("1706.03762v3.pdf")
        and getattr(p, "mime_type", None) == "application/pdf"
        for p in shared
    )
