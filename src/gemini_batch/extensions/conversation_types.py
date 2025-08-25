from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Exchange:
    user: str
    assistant: str
    error: bool
    # optional audit fields surfaced by core Result/metrics (do not compute here)
    estimate_min: int | None = None
    estimate_max: int | None = None
    actual_tokens: int | None = None
    in_range: bool | None = None
    warnings: tuple[str, ...] = ()

@dataclass(frozen=True)
class ConversationPolicy:
    keep_last_n: int | None = None
    widen_max_factor: float | None = None
    clamp_max_tokens: int | None = None
    prefer_json_array: bool = False
    execution_cache_name: str | None = None
    reuse_cache_only: bool = False  # intent; provider capability decides behavior

@dataclass(frozen=True)
class PromptSet:
    prompts: tuple[str, ...]
    mode: Literal["single","sequential","vectorized"] = "single"

@dataclass(frozen=True)
class ConversationState:
    sources: tuple[Any, ...]
    turns: tuple[Exchange, ...]
    cache_key: str | None = None
    cache_artifacts: tuple[str, ...] = ()
    cache_ttl_seconds: int | None = None
    policy: ConversationPolicy | None = None
    version: int = 0

@dataclass(frozen=True)
class BatchMetrics:
    per_prompt: tuple[dict[str, int | float], ...]
    totals: dict[str, int | float]

@dataclass(frozen=True)
class ConversationAnalytics:
    total_turns: int
    error_turns: int
    success_rate: float
    total_estimated_tokens: int | None = None
    total_actual_tokens: int | None = None
    estimation_accuracy: float | None = None
    avg_response_length: float = 0.0
    total_user_chars: int = 0
    total_assistant_chars: int = 0
