import os
from pathlib import Path
from typing import Any

import pytest

from gemini_batch.config import GeminiConfig
from gemini_batch.core.types import (
    APICall,
    ExecutionPlan,
    ExplicitCachePlan,
    FileRefPart,
    InitialCommand,
    PlannedCommand,
    ResolvedCommand,
    Success,
    TextPart,
    UploadTask,
)
from gemini_batch.pipeline.api_handler import APIHandler


def _mk_planned_with_upload_and_cache(tmp_path: Path) -> PlannedCommand:
    # Primary parts: placeholder for file at index 0, then a text prompt
    placeholder = FileRefPart(uri="", mime_type="text/plain")
    initial = InitialCommand(
        sources=("s",), prompts=("q",), config=GeminiConfig(api_key="k")
    )
    resolved = ResolvedCommand(initial=initial, resolved_sources=())
    primary = APICall(
        model_name="gemini-2.0-flash",
        api_parts=(placeholder, TextPart(text="Hello")),
        api_config={"system_instruction": "You are helpful."},
    )
    up = UploadTask(part_index=0, local_path=tmp_path / "f.txt", mime_type="text/plain")
    plan = ExecutionPlan(
        primary_call=primary,
        upload_tasks=(up,),
        explicit_cache=ExplicitCachePlan(create=True, contents_part_indexes=(0,)),
    )
    return PlannedCommand(resolved=resolved, execution_plan=plan)


@pytest.mark.asyncio
async def test_upload_substitution_and_cache_create_mock_path(tmp_path):
    # Ensure file exists for upload path
    (tmp_path / "f.txt").write_text("data", encoding="utf-8")

    handler = APIHandler()
    planned = _mk_planned_with_upload_and_cache(tmp_path)
    # Mock path is default; should not error even though adapter methods are mocked
    result = await handler.handle(planned)
    assert isinstance(result, Success)


class _FakeAdapterCapture:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.created_cache_names: list[tuple[str, int | None]] = []

    async def upload_file_local(
        self, path: os.PathLike[str] | str, mime_type: str | None
    ) -> Any:
        return {"uri": f"files/mock/{os.fspath(path)}", "mime_type": mime_type}

    async def create_cache(
        self,
        *,
        model_name: str,
        content_parts: tuple[Any, ...],  # noqa: ARG002
        system_instruction: str | None,  # noqa: ARG002
        ttl_seconds: int | None,
    ) -> str:
        self.created_cache_names.append((model_name, ttl_seconds))
        return "cachedContents/fake-123"

    async def generate(
        self,
        *,
        model_name: str,
        api_parts: tuple[Any, ...],  # noqa: ARG002
        api_config: dict[str, object],
    ) -> dict[str, Any]:
        self.calls.append({"model": model_name, "cfg": dict(api_config)})
        return {"text": "ok", "model": model_name, "usage": {"total_token_count": 10}}


@pytest.mark.asyncio
async def test_cache_create_and_use_real_path(tmp_path):
    (tmp_path / "f.txt").write_text("data", encoding="utf-8")
    fake = _FakeAdapterCapture()
    handler = APIHandler(adapter=fake)
    planned = _mk_planned_with_upload_and_cache(tmp_path)
    # Attach ttl to plan
    ec = planned.execution_plan.explicit_cache
    assert ec is not None
    object.__setattr__(
        ec, "ttl_seconds", 3600
    )  # dataclass frozen workaround not ideal in real code

    result = await handler.handle(planned)
    assert isinstance(result, Success)
    # Ensure cached_content was included in generate config
    assert any("cached_content" in c["cfg"] for c in fake.calls)
    # Ensure ttl propagated to create_cache
    assert fake.created_cache_names and fake.created_cache_names[0][1] == 3600


class _FlakyAdapter:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.first = True

    async def upload_file_local(
        self,
        path: os.PathLike[str] | str,
        mime_type: str | None,  # noqa: ARG002
    ) -> Any:
        return {"uri": f"files/mock/{os.fspath(path)}"}

    async def create_cache(self, **_: Any) -> str:
        return "cachedContents/fake-xyz"

    async def generate(
        self,
        *,
        model_name: str,
        api_parts: tuple[Any, ...],  # noqa: ARG002
        api_config: dict[str, object],
    ) -> dict[str, Any]:
        # Fail on first attempt if cached_content present; succeed when retried without it
        if self.first and "cached_content" in api_config:
            self.first = False
            raise RuntimeError("cache too large")
        return {"text": "ok", "model": model_name, "usage": {"total_token_count": 5}}


@pytest.mark.asyncio
async def test_retry_without_cache_on_generate_error(tmp_path):
    (tmp_path / "f.txt").write_text("data", encoding="utf-8")
    handler = APIHandler(adapter=_FlakyAdapter())
    planned = _mk_planned_with_upload_and_cache(tmp_path)
    result = await handler.handle(planned)
    assert isinstance(result, Success)
