"""Scenario-first convenience helpers for common operations.

These functions provide a minimal "pit of success" entrypoint over the
underlying executor and command pipeline, without changing core behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemini_batch.config import FrozenConfig, resolve_config
from gemini_batch.core.types import InitialCommand, Source
from gemini_batch.executor import GeminiExecutor, create_executor
from gemini_batch.pipeline.hints import CachePolicyHint, ResultHint

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gemini_batch.types import ResultEnvelope

if TYPE_CHECKING:  # import for typing only
    from collections.abc import Iterable


async def run_simple(
    prompt: str,
    *,
    source: Source | None = None,
    cfg: FrozenConfig | None = None,
    prefer_json: bool = False,
) -> ResultEnvelope:
    """Run a simple query (optionally RAG on a single source).

    Args:
        prompt: The user prompt to execute.
        source: A single explicit `types.Source`. Use `types.Source.from_text()`
            for text content or `types.Source.from_file()` for files. Strings
            are not accepted directly to avoid ambiguity.
        cfg: Optional frozen configuration. If omitted, `resolve_config()` is used.
        prefer_json: Hint the extractor to prefer JSON array when reasonable.

    Returns:
        Result envelope dictionary.

    Example:
        ```python
        from gemini_batch import types

        # Simple text analysis
        result = await run_simple(
            "What is the main theme?",
            source=types.Source.from_text("Long text..."),
        )

        # File analysis
        result = await run_simple(
            "Summarize this document",
            source=types.Source.from_file("report.pdf"),
        )
        ```

    See Also:
        For advanced control, use `GeminiExecutor` directly.
        For multiple prompts, use `run_batch()`.
    """
    final_cfg = cfg or resolve_config()
    executor: GeminiExecutor = create_executor(final_cfg)

    sources: tuple[Source, ...] = (source,) if source is not None else ()

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
    sources: Iterable[Source] = (),
    *,
    cfg: FrozenConfig | None = None,
    prefer_json: bool = False,
    enable_caching: bool | None = None,
    min_tokens_floor: int | None = None,
) -> ResultEnvelope:
    """Run multiple prompts over one or many sources efficiently.

    Covers multi-question analysis, complex synthesis, and parallel batch by
    relying on the planner's vectorization of prompts with shared context.

    Args:
        prompts: One or more user prompts.
        sources: Zero or more explicit `types.Source` objects. Use
            `types.Source.from_text()` for text content or
            `types.Source.from_file()` for files. Strings are not accepted
            directly to avoid ambiguity. For directories, use the explicit
            helper `types.sources_from_directory(path)`.
        cfg: Optional frozen configuration; resolved if omitted.
        prefer_json: Hint extractor to prefer JSON array when reasonable.
        enable_caching: Optional override for cache enablement behavior.
            - True: Force-enables caching for this call.
            - False: Force-disables caching for this call.
            - None (default): Use the value from resolved configuration
              (e.g., pyproject.toml or environment variables).
            Note: Caching may be enabled by default depending on your project
            configuration. Set to False to be certain.
        min_tokens_floor: Optional cache policy floor override in tokens.

    Returns:
        Result envelope dictionary with answers ordered by prompts.

    Example:
        ```python
        from gemini_batch import types

        # Multi-question analysis
        questions = ["What are the key themes?", "Who are the main characters?"]
        result = await run_batch(
            questions, sources=[types.Source.from_file("story.txt")]
        )

        # Batch processing with caching
        result = await run_batch(
            prompts=["Analyze tone", "Extract quotes"],
            sources=[types.Source.from_file("large_document.pdf")],
            enable_caching=True,
        )

        # Directory expansion (explicit helper)
        result = await run_batch(
            prompts=["Index files"],
            sources=types.sources_from_directory("docs/"),
        )
        ```

    See Also:
        For single prompts, use `run_simple()`.
        For advanced pipeline control, use `GeminiExecutor` with `InitialCommand`.
    """
    final_cfg = cfg or resolve_config()
    executor: GeminiExecutor = create_executor(final_cfg)

    # No implicit path detection; callers must pass explicit Sources
    resolved_sources: list[Source] = list(sources)

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
