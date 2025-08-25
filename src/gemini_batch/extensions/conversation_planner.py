from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from gemini_batch.core.types import ConversationTurn
from gemini_batch.pipeline.hints import (
    CacheHint,
    EstimationOverrideHint,
    ExecutionCacheName,
    ResultHint,
)

if TYPE_CHECKING:
    from .conversation_types import ConversationPolicy, ConversationState, PromptSet


@dataclass(frozen=True)
class ConversationPlan:
    sources: tuple[Any, ...]
    history: tuple[ConversationTurn, ...]
    prompts: tuple[str, ...]
    strategy: Literal["sequential", "vectorized"]
    hints: tuple[object, ...]


def compile_conversation(
    state: ConversationState, prompt_set: PromptSet, policy: ConversationPolicy | None
) -> ConversationPlan:
    # history window
    full = tuple(ConversationTurn(q.user, q.assistant, q.error) for q in state.turns)
    hist = (
        full[-policy.keep_last_n :]
        if (policy and policy.keep_last_n and policy.keep_last_n > 0)
        else full
    )

    # hints
    hints: list[object] = []
    if state.cache_key:
        hints.append(
            CacheHint(
                deterministic_key=state.cache_key,
                artifacts=state.cache_artifacts,
                ttl_seconds=state.cache_ttl_seconds,
                reuse_only=bool(policy and policy.reuse_cache_only),
            )
        )
    if policy and (policy.widen_max_factor or policy.clamp_max_tokens):
        hints.append(
            EstimationOverrideHint(
                widen_max_factor=policy.widen_max_factor or 1.0,
                clamp_max_tokens=policy.clamp_max_tokens,
            )
        )
    if policy and policy.prefer_json_array:
        hints.append(ResultHint(prefer_json_array=True))
    if policy and policy.execution_cache_name:
        hints.append(ExecutionCacheName(cache_name=policy.execution_cache_name))

    # strategy (data, not branches elsewhere)
    strategy = cast(
        "Literal['sequential', 'vectorized']",
        "vectorized" if (prompt_set.mode == "vectorized") else "sequential",
    )
    return ConversationPlan(
        sources=state.sources,
        history=hist,
        prompts=prompt_set.prompts,
        strategy=strategy,
        hints=tuple(hints),
    )
