import os
from typing import Any

import pytest

from gemini_batch.config import resolve_config
from gemini_batch.core.types import (
    APICall,
    ExecutionPlan,
    ExplicitCachePlan,
    InitialCommand,
    PlannedCommand,
    ResolvedCommand,
    Success,
    TextPart,
    UploadTask,
)
from gemini_batch.pipeline.api_handler import APIHandler
from gemini_batch.pipeline.registries import CacheRegistry, FileRegistry


class _SpyAdapter:
    def __init__(self) -> None:
        self.uploads: list[str] = []
        self.creates: list[str] = []
        self.generations: list[dict[str, Any]] = []

    async def upload_file_local(
        self, path: os.PathLike[str] | str, _mime_type: str | None
    ) -> Any:
        self.uploads.append(os.fspath(path))
        return {"uri": f"files/mock/{os.fspath(path)}"}

    async def create_cache(
        self,
        *,
        model_name: str,
        content_parts: tuple[Any, ...],  # noqa: ARG002
        system_instruction: str | None,  # noqa: ARG002
        ttl_seconds: int | None,  # noqa: ARG002
    ) -> str:
        self.creates.append(model_name)
        return "cachedContents/test-1"

    async def generate(
        self,
        *,
        model_name: str,
        api_parts: tuple[Any, ...],  # noqa: ARG002
        api_config: dict[str, object],
    ) -> dict[str, Any]:
        self.generations.append({"model": model_name, "cfg": dict(api_config)})
        return {"text": "ok", "model": model_name, "usage": {"total_token_count": 1}}


@pytest.mark.asyncio
async def test_handler_uses_registries_for_upload_and_cache(tmp_path):
    # Prepare planned command with upload and explicit cache
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    initial = InitialCommand(
        sources=("s",),
        prompts=("p",),
        config=resolve_config(overrides={"api_key": "k"}),
    )
    resolved = ResolvedCommand(initial=initial, resolved_sources=())
    primary = APICall(
        model_name="gemini-2.0-flash",
        api_parts=(TextPart(text="p"),),
        api_config={"system_instruction": "si"},
    )
    up = UploadTask(part_index=0, local_path=tmp_path / "f.txt", required=False)
    exp = ExplicitCachePlan(
        create=True, contents_part_indexes=(0,), deterministic_key="key1"
    )
    plan = ExecutionPlan(primary_call=primary, upload_tasks=(up,), explicit_cache=exp)
    planned = PlannedCommand(resolved=resolved, execution_plan=plan)

    # Spy adapter injected explicitly
    spy = _SpyAdapter()

    # Registries start empty
    cache_reg = CacheRegistry()
    file_reg = FileRegistry()
    handler = APIHandler(
        registries={"cache": cache_reg, "files": file_reg}, adapter=spy
    )

    # First call: should upload and create cache
    r1 = await handler.handle(planned)
    assert isinstance(r1, Success)
    assert spy.uploads  # uploaded once
    assert spy.creates  # created cache

    # Registry populated
    assert cache_reg.get("key1") == "cachedContents/test-1"
    assert file_reg.get(os.fspath(tmp_path / "f.txt")) is not None

    # Second call: should hit registries; no new upload or create
    spy.uploads.clear()
    spy.creates.clear()
    r2 = await handler.handle(planned)
    assert isinstance(r2, Success)
    assert spy.uploads == []
    assert spy.creates == []
