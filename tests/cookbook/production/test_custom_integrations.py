from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import pytest

from tests.cookbook.support import load_recipe_module

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.unit
@pytest.mark.cookbook
def test_custom_integrations_prints_status_and_timing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    patch_run_batch: Callable[..., None],
    make_env: Callable[..., dict[str, Any]],
) -> None:
    # Arrange files
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")

    mod: Any = load_recipe_module("cookbook/production/custom-integrations.py")
    main_async = mod.main_async

    async def fake_run_batch(*_args, **_kwargs):
        return make_env(status="ok", metrics={})

    # A minimal fake TelemetryContext that triggers reporter outputs
    class FakeCtx:
        def __init__(self, reporter: Any) -> None:
            self.reporter = reporter

        def __call__(self, _name: str) -> FakeCtx:
            return self

        def __enter__(self) -> Self:
            # Emit a fake timing on enter
            self.reporter.record_timing(
                "cookbook.custom_integrations.run", 0.001, depth=1
            )
            self.reporter.record_metric(
                "cookbook.custom_integrations.metric", 1, parent_scope=None
            )
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object,
        ) -> Literal[False]:
            return False

    def fake_telemetry_context(reporter: Any) -> FakeCtx:
        return FakeCtx(reporter)

    mod.TelemetryContext = fake_telemetry_context
    patch_run_batch(mod, fake_run_batch)

    # Act
    asyncio.run(main_async(Path(tmp_path)))

    # Assert
    out = capsys.readouterr().out
    assert "Status: ok" in out
    assert "TIMING" in out  # reporter prints
    assert "METRIC" in out
