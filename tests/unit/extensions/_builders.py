"""Lightweight builders for conversation extension tests.

These helpers keep tests focused on behavior by providing minimal,
typed construction of core extension data structures.
"""

from __future__ import annotations

from typing import Any

from gemini_batch.extensions.conversation_types import (
    ConversationPolicy,
    ConversationState,
    PromptSet,
)


def make_state(
    *,
    sources: tuple[Any, ...] | None = None,
    turns: tuple[Any, ...] | None = None,
    cache_key: str | None = None,
    artifacts: tuple[str, ...] | None = None,
    ttl_seconds: int | None = None,
    policy: ConversationPolicy | None = None,
    version: int = 0,
) -> ConversationState:
    """Construct a `ConversationState` with sensible defaults."""
    return ConversationState(
        sources=tuple(sources or ()),
        turns=tuple(turns or ()),
        cache_key=cache_key,
        cache_artifacts=tuple(artifacts or ()),
        cache_ttl_seconds=ttl_seconds,
        policy=policy,
        version=version,
    )


def make_policy(**overrides: Any) -> ConversationPolicy:
    """Construct a `ConversationPolicy` applying any field overrides."""
    base = ConversationPolicy()
    return ConversationPolicy(**{**base.__dict__, **overrides})


def make_prompt_set(mode: str, *prompts: str) -> PromptSet:
    """Create a `PromptSet` for a given mode name.

    Args:
        mode: One of "single", "sequential", or "vectorized".
        prompts: Prompt strings.
    """
    mode = mode.lower().strip()
    if mode == "single":
        return PromptSet.single(prompts[0] if prompts else "")
    if mode == "sequential":
        return PromptSet.sequential(*prompts)
    if mode == "vectorized":
        return PromptSet.vectorized(*prompts)
    raise ValueError(f"Unknown mode: {mode}")
