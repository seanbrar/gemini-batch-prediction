"""Hint Capsules for extension-to-core communication.

Precedence (cache)
- Planning: `CacheHint` influences the planner's `ExplicitCachePlan` and
  optional `APICall.cache_name_to_use`. Best-effort; unknown hints are ignored.
- Execution: `ExecutionCacheName` overrides the cache name used for the API
  attempt. Best-effort; does not mutate the plan.

This module defines immutable hint types that extensions can use to express
intent to the pipeline without coupling the core to domain-specific logic.
Hints are consumed by pipeline stages in a fail-soft manner — unknown or
unsupported hints are silently ignored. If multiple hints of the same class
are supplied, the first one encountered is used ("first one wins"). For
boolean preference-style hints (e.g., ResultHint), the presence of any
instance enabling the preference is sufficient.

Design principles:
- Immutable dataclasses (frozen=True)
- Optional and fail-soft - system works identically when hints are None/empty
- Provider-neutral - no provider-specific details in core pipeline
- Data-driven preferences, not control flow changes
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclasses.dataclass(frozen=True, slots=True)
class CacheHint:
    """Hint for deterministic cache identity and policy knobs.

    Used by the Execution Planner to override caching decisions and provide
    explicit cache keys. Supports both create-new and reuse-only policies.

    Attributes:
        deterministic_key: Explicit cache key to use instead of planner-generated key
        artifacts: Optional tuple of artifact identifiers for cache metadata
        ttl_seconds: Optional TTL override for cache entry lifetime
        reuse_only: If True, only reuse existing cache, don't create new entries
    """

    deterministic_key: str
    artifacts: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    ttl_seconds: int | None = None
    reuse_only: bool = False

    def __post_init__(self) -> None:
        """Validate basic invariants for cache hints."""
        # Enforce simple, explicit invariants to reduce downstream checks
        key = (self.deterministic_key or "").strip()
        if not key:
            raise ValueError("CacheHint.deterministic_key must be a non-empty string")
        if self.ttl_seconds is not None and self.ttl_seconds < 0:
            raise ValueError("CacheHint.ttl_seconds must be >= 0 when provided")

        # Artifacts are optional metadata; if supplied, ensure all are non-empty strings
        def _all_non_empty_str(items: Iterable[str]) -> bool:
            return all(isinstance(a, str) and bool(a.strip()) for a in items)

        if self.artifacts and not _all_non_empty_str(self.artifacts):
            raise ValueError("CacheHint.artifacts must contain only non-empty strings")


@dataclasses.dataclass(frozen=True, slots=True)
class EstimationOverrideHint:
    """Hint for conservative adjustments to token estimates.

    Used by the Execution Planner to apply planner-scoped transforms to
    token estimates without introducing provider coupling. Semantics are
    conservative and explicitly bounded: the planner widens ``max_tokens``
    by ``widen_max_factor`` and then optionally clamps it to
    ``clamp_max_tokens``; invariants ``max_tokens >= min_tokens`` and
    ``expected_tokens <= max_tokens`` are enforced.

    Attributes:
        widen_max_factor: Multiply max_tokens by this factor (default 1.0 = no change)
        clamp_max_tokens: Optional upper bound to clamp max_tokens after widening
    """

    widen_max_factor: float = 1.0
    clamp_max_tokens: int | None = None

    def __post_init__(self) -> None:
        """Validate conservative override parameters."""
        # Keep override semantics conservative and explicit
        if not math.isfinite(self.widen_max_factor) or self.widen_max_factor < 1.0:
            raise ValueError(
                "EstimationOverrideHint.widen_max_factor must be finite and >= 1.0"
            )
        if self.clamp_max_tokens is not None and self.clamp_max_tokens < 0:
            raise ValueError(
                "EstimationOverrideHint.clamp_max_tokens must be >= 0 when provided"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class ResultHint:
    """Hint for non-breaking transform preferences in extraction.

    Used by the Result Builder to bias transform order while preserving
    Tier-2 fallback guarantees. Never causes extraction to fail.

    Attributes:
        prefer_json_array: If True, bias toward json_array transform in Tier-1
    """

    prefer_json_array: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionCacheName:
    """Hint for execution-time cache name override.

    Used by the API Handler to override cache names at execution time.
    This is optional and best-effort - failures are silently ignored.

    Attributes:
        cache_name: The cache name to use for API execution
    """

    cache_name: str

    def __post_init__(self) -> None:
        """Validate cache name for execution-time override."""
        if not (self.cache_name or "").strip():
            raise ValueError("ExecutionCacheName.cache_name must be a non-empty string")


# Union type for all supported hints (public capsule surface)
type Hint = CacheHint | EstimationOverrideHint | ResultHint | ExecutionCacheName
