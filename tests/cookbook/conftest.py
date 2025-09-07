from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

# Ensure repository root and `src/` are importable for cookbook modules
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
for p in (str(_ROOT), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


# --- Tiny, reusable cookbook helpers ---


@pytest.fixture
def make_files(tmp_path: Path) -> Callable[[Iterable[str], str | bytes], list[Path]]:
    """Create files under a temp directory.

    Returns a function taking (names, content) and yielding the created Paths.
    Content may be text or bytes; uses write_text for str and write_bytes for bytes.
    """

    def _make(names: Iterable[str], content: str | bytes = "x") -> list[Path]:
        out: list[Path] = []
        for name in names:
            p = tmp_path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                p.write_bytes(content)
            else:
                p.write_text(content)
            out.append(p)
        return out

    return _make


@pytest.fixture
def patch_run_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Any, dict[str, Any] | Callable[..., Awaitable[dict[str, Any]]]], None]:
    """Patch a module's run_batch attribute with a simple async stub or custom callable.

    Usage: patch_run_batch(mod, payload_dict) or patch_run_batch(mod, async_fn)
    """

    def _patch(
        mod: Any, payload: dict[str, Any] | Callable[..., Awaitable[dict[str, Any]]]
    ) -> None:
        if callable(payload):
            fn = payload
        else:

            async def fn(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
                return payload

        monkeypatch.setattr(mod, "run_batch", fn)

    return _patch


@pytest.fixture
def patch_run_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Any, dict[str, Any] | Callable[..., Awaitable[dict[str, Any]]]], None]:
    """Patch a module's run_parallel attribute similarly to patch_run_batch."""

    def _patch(
        mod: Any, payload: dict[str, Any] | Callable[..., Awaitable[dict[str, Any]]]
    ) -> None:
        if callable(payload):
            fn = payload
        else:

            async def fn(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
                return payload

        monkeypatch.setattr(mod, "run_parallel", fn)

    return _patch


@pytest.fixture
def cap_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Callable[[Any, list[str], Callable[[], Any]], tuple[str, str]]:
    """Invoke a CLI entry with argv and capture stdout/stderr.

    The first argument accepts the target module for ergonomic symmetry with test
    call sites; it is not used internally. Pass the callable to execute (usually
    ``module.main``). Returns a pair ``(stdout, stderr)``.
    """

    def _invoke(
        _module: Any, argv: list[str], call: Callable[[], Any]
    ) -> tuple[str, str]:
        import sys as _sys

        monkeypatch.setattr(_sys, "argv", argv)
        call()
        captured = capsys.readouterr()
        return captured.out, captured.err

    return _invoke


@pytest.fixture
def make_env() -> Callable[..., dict[str, Any]]:
    """Factory for building recipe env/envelope dicts consistently.

    Args:
        status: Optional status string (e.g., "ok", "partial").
        answers: Optional list of answers payloads.
        usage: Optional mapping for token usage (merged with defaults).
        metrics: Optional mapping for metrics (merged with defaults).

    Returns:
        Dict with keys present only when provided or non-empty.
    """

    def _make(
        *,
        status: str | None = None,
        answers: list[Any] | None = None,
        usage: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if status is not None:
            out["status"] = status
        if answers is not None:
            out["answers"] = answers
        if usage:
            out["usage"] = usage
        if metrics:
            out["metrics"] = metrics
        return out

    return _make
