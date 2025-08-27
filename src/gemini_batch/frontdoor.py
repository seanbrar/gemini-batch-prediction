"""Scenario-first convenience helpers for common operations.

These functions provide a minimal "pit of success" entrypoint over the
underlying executor and command pipeline, without changing core behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from gemini_batch.config import FrozenConfig, resolve_config
from gemini_batch.core.types import InitialCommand, Source
from gemini_batch.executor import GeminiExecutor, create_executor
from gemini_batch.pipeline.hints import CachePolicyHint, ResultHint

if TYPE_CHECKING:  # import for typing only
    from collections.abc import Iterable


async def run_simple(
    prompt: str,
    *,
    source: str | Path | None = None,
    cfg: FrozenConfig | None = None,
    prefer_json: bool = False,
) -> dict[str, Any]:
    """Run a simple query (optionally RAG on a single source).

    Args:
        prompt: The user prompt to execute.
        source: Optional single source (text or path). Strings that resolve to
            existing paths will be treated as files; otherwise treated as text.
        cfg: Optional frozen configuration. If omitted, `resolve_config()` is used.
        prefer_json: Hint the extractor to prefer JSON array when reasonable.

    Returns:
        Result envelope dictionary.

    Example:
        ```python
        # Simple text analysis
        result = await run_simple("What is the main theme?", source="Long text...")

        # File analysis
        result = await run_simple("Summarize this document", source="report.pdf")
        ```

    See Also:
        For advanced control, use `GeminiExecutor` directly.
        For multiple prompts, use `run_batch()`.
    """
    final_cfg = cfg or resolve_config()
    executor: GeminiExecutor = create_executor(final_cfg)

    sources: tuple[Any, ...]
    if source is None:
        sources = ()
    elif isinstance(source, Path) or Path(str(source)).exists():
        sources = (Source.from_file(str(source)),)
    else:
        sources = (Source.from_text(str(source)),)

    hints: list[object] = []
    if prefer_json:
        hints.append(ResultHint(prefer_json_array=True))

    cmd = InitialCommand(
        sources=sources,
        prompts=(str(prompt),),
        config=executor.config,
        hints=tuple(hints) or None,
    )
    return await executor.execute(cmd)


async def run_batch(
    prompts: Iterable[str],
    sources: Iterable[str | Path] = (),
    *,
    cfg: FrozenConfig | None = None,
    prefer_json: bool = False,
    enable_caching: bool | None = None,
    min_tokens_floor: int | None = None,
) -> dict[str, Any]:
    """Run multiple prompts over one or many sources efficiently.

    Covers multi-question analysis, complex synthesis, and parallel batch by
    relying on the planner's vectorization of prompts with shared context.

    Args:
        prompts: One or more user prompts.
        sources: Zero or more sources (text or file paths). Paths are detected.
        cfg: Optional frozen configuration; resolved if omitted.
        prefer_json: Hint extractor to prefer JSON array when reasonable.
        enable_caching: Optional override for cache enablement behavior. When
            True and provider supports caching, a cache may be created for
            shared context. If None, uses config.
        min_tokens_floor: Optional cache policy floor override in tokens.

    Returns:
        Result envelope dictionary with answers ordered by prompts.

    Example:
        ```python
        # Multi-question analysis
        questions = ["What are the key themes?", "Who are the main characters?"]
        result = await run_batch(questions, sources=["story.txt"])

        # Batch processing with caching
        result = await run_batch(
            prompts=["Analyze tone", "Extract quotes"],
            sources=["large_document.pdf"],
            enable_caching=True,
        )
        ```

    See Also:
        For single prompts, use `run_simple()`.
        For advanced pipeline control, use `GeminiExecutor` with `InitialCommand`.
    """
    final_cfg = cfg or resolve_config()
    executor: GeminiExecutor = create_executor(final_cfg)

    # Resolve sources eagerly using ergonomic constructors
    resolved_sources: list[Any] = []
    for s in sources:
        ps = Path(str(s))
        if ps.exists():
            resolved_sources.append(Source.from_file(ps))
        else:
            resolved_sources.append(Source.from_text(str(s)))

    hints: list[object] = []
    if prefer_json:
        hints.append(ResultHint(prefer_json_array=True))
    if enable_caching is True or min_tokens_floor is not None:
        hints.append(
            CachePolicyHint(
                first_turn_only=True,
                # Respect floors; allow override when provided
                respect_floor=True,
                min_tokens_floor=min_tokens_floor,
            )
        )

    cmd = InitialCommand.strict(
        sources=tuple(resolved_sources),
        prompts=tuple(str(p) for p in prompts),
        config=executor.config,
        hints=tuple(hints) or None,
    )
    return await executor.execute(cmd)
