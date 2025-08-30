"""Cache stage: capability-aware, execution-time cache resolution.

This stage centralizes all cache decisions at execution time, following the
capability-based provider abstraction. It:

- Runs only when the provider supports explicit caching (duck-typed via
  CachingCapability)
- Resolves cache names using a registry keyed by deterministic identities
- Creates caches when allowed (respecting reuse-only hints and TTL)
- Updates the execution plan with `cache_name_to_use` for downstream generation

Planner remains pure and does not make cache decisions.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, cast

from gemini_batch.constants import INLINE_CACHE_MAX_BYTES
from gemini_batch.core.exceptions import APIError
from gemini_batch.core.models import get_model_capabilities
from gemini_batch.core.types import (
    APICall,
    ExecutionPlan,
    Failure,
    PlannedCommand,
    Result,
    Success,
    TokenEstimate,
)
from gemini_batch.pipeline.adapters.base import CachingCapability, GenerationAdapter
from gemini_batch.pipeline.base import BaseAsyncHandler
from gemini_batch.pipeline.cache_identity import det_shared_key
from gemini_batch.pipeline.hints import CacheHint, CachePolicyHint

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemini_batch.pipeline.registries import SimpleRegistry

logger = logging.getLogger(__name__)


class CacheStage(BaseAsyncHandler[PlannedCommand, PlannedCommand, APIError]):
    """Execution-time cache handler.

    Decides whether to reuse or create caches and annotates APICalls with the
    `cache_name_to_use`. No-ops when provider lacks caching capability.
    """

    def __init__(
        self,
        *,
        registries: dict[str, SimpleRegistry] | None = None,
        adapter_factory: Callable[[str], GenerationAdapter] | None = None,
    ) -> None:
        """Initialize cache stage with registry and adapter factory.

        Args:
            registries: Optional registries dictionary containing cache registry.
            adapter_factory: Optional factory for creating generation adapters.
        """
        self._cache_registry = (registries or {}).get("cache")
        self._adapter_factory = adapter_factory

    async def handle(self, command: PlannedCommand) -> Result[PlannedCommand, APIError]:
        """Process command to resolve or create cache and update execution plan.

        Args:
            command: Planned command containing execution plan and resolved data.

        Returns:
            Updated command with cache applied to execution plan, or error result.
        """
        try:
            initial = command.resolved.initial
            cfg = initial.config

            # Select adapter only when real API is enabled; otherwise no-op.
            adapter = _select_caching_adapter(cfg, self._adapter_factory)
            if adapter is None or not isinstance(adapter, CachingCapability):
                return Success(command)

            plan = command.execution_plan

            # Read hints once (fail-soft) and pass down
            hints = tuple(getattr(initial, "hints", ()) or ())
            cache_hint = next((h for h in hints if isinstance(h, CacheHint)), None)
            policy_hint = next(
                (h for h in hints if isinstance(h, CachePolicyHint)), None
            )

            # Resolve or create a cache for shared parts (vectorized or single)
            cache_name = await self._resolve_or_create_cache_name(
                adapter=adapter,
                command=command,
                cache_hint=cache_hint,
                policy_hint=policy_hint,
            )

            if not cache_name:
                return Success(command)

            # Apply cache name to the plan and preserve all other fields
            updated_plan = _apply_cache_to_plan(plan, cache_name)
            updated_command = dataclasses.replace(command, execution_plan=updated_plan)
            return Success(updated_command)
        except APIError as e:
            return Failure(e)
        except Exception as e:
            return Failure(APIError(f"Cache stage failed: {e}"))

    # --- Internal helpers ---
    async def _resolve_or_create_cache_name(
        self,
        *,
        adapter: CachingCapability,
        command: PlannedCommand,
        cache_hint: CacheHint | None,
        policy_hint: CachePolicyHint | None,
    ) -> str | None:
        """Resolve or create a cache name for shared parts across both paths.

        Uses shared_parts from the plan to identify the shared context (history/files).
        Reuses registry entries when present; otherwise creates a cache if policy allows.
        """
        plan = command.execution_plan
        # Use authoritative calls set; derived primary retained for back-compat
        call0 = plan.calls[0]
        model_name = call0.model_name
        system_instruction = cast(
            "str | None", call0.api_config.get("system_instruction")
        )

        # Inline file placeholders for caching payload via a dedicated shaper
        shared_parts = list(getattr(plan, "shared_parts", ()) or ())
        inline_parts = _shape_cache_payload(shared_parts)
        if not inline_parts:
            return None

        initial = command.resolved.initial
        cfg = initial.config
        if not cfg.enable_caching and cache_hint is None:
            return None

        reg_key = _compute_cache_key(
            model_name=model_name,
            system_instruction=system_instruction,
            command=command,
            cache_hint=cache_hint,
        )

        # Reuse from registry if available
        cached = _registry_get(self._cache_registry, reg_key)
        if isinstance(cached, str):
            return cached

        if cache_hint and cache_hint.reuse_only:
            return None

        # Apply cache policy checks
        policy_decision = _resolve_cache_policy_decision(
            model_name=model_name,
            policy_hint=policy_hint,
            token_estimate=command.token_estimate,
            first_turn=_is_first_turn(initial),
            enable_caching=cfg.enable_caching,
        )
        if not policy_decision:
            return None

        ttl = cache_hint.ttl_seconds if cache_hint else cfg.ttl_seconds
        try:
            created = await adapter.create_cache(
                model_name=model_name,
                content_parts=tuple(inline_parts),
                system_instruction=system_instruction,
                ttl_seconds=ttl,
            )
        except Exception as e:
            logger.debug("create_cache failed: %s", e, exc_info=True)
            return None

        if isinstance(created, str):
            _registry_set(self._cache_registry, reg_key, created)
            return created
        return None


# --- Small helpers ---


def _inline_file_placeholders(parts: list[Any]) -> list[Any]:
    inline: list[Any] = []
    from gemini_batch.core.types import FileInlinePart, FilePlaceholder

    for p in parts:
        if isinstance(p, FilePlaceholder):
            try:
                # Skip inlining very large files to avoid memory spikes
                try:
                    size = p.local_path.stat().st_size
                except Exception:
                    size = None
                if size is not None and size > INLINE_CACHE_MAX_BYTES:
                    logger.debug(
                        "Skipping inline caching for large file: %s (%d bytes > %d limit)",
                        p.local_path.name,
                        size,
                        INLINE_CACHE_MAX_BYTES,
                    )
                    inline.append(p)
                    continue

                data = p.local_path.read_bytes()
                inline.append(
                    FileInlinePart(
                        mime_type=p.mime_type or "application/octet-stream",
                        data=data,
                    )
                )
            except Exception:
                logger.debug("skipping unreadable file for caching purposes: %s", p)
                continue
        else:
            inline.append(p)
    return inline


def _shape_cache_payload(parts: list[Any]) -> list[Any]:
    """Shape shared parts into a cache-friendly payload.

    Currently, this inlines file placeholders up to a size threshold.
    Future strategies (e.g., references) can be added here without touching
    policy or identity computation.
    """
    return _inline_file_placeholders(parts)


def _compute_cache_key(
    *,
    model_name: str,
    system_instruction: str | None,
    command: PlannedCommand,
    cache_hint: CacheHint | None,
) -> str:
    """Compute deterministic cache key from hint or shared-context identity."""
    if cache_hint and (cache_hint.deterministic_key or "").strip():
        return cache_hint.deterministic_key
    return det_shared_key(model_name, system_instruction, command)


def _with_cache_name(call: APICall, cache_name: str) -> APICall:
    return APICall(
        model_name=call.model_name,
        api_parts=call.api_parts,
        api_config=dict(call.api_config),
        cache_name_to_use=cache_name,
    )


def _apply_cache_to_plan(plan: ExecutionPlan, cache_name: str) -> ExecutionPlan:
    """Return a copy of plan with cache applied to vectorized or single call.

    Preserves all other fields (fallback, rate constraints, uploads).
    """
    updated_calls = tuple(_with_cache_name(c, cache_name) for c in plan.calls)
    return dataclasses.replace(plan, calls=updated_calls)


def _registry_get(reg: Any, key: str) -> Any | None:
    if reg is None:
        return None
    try:
        return reg.get(key)
    except Exception:
        return None


def _registry_set(reg: Any, key: str, value: Any) -> None:
    if reg is None:
        return
    from contextlib import suppress

    with suppress(Exception):
        reg.set(key, value)


def _resolve_cache_policy_decision(
    *,
    model_name: str,
    policy_hint: CachePolicyHint | None,
    token_estimate: TokenEstimate | None,
    first_turn: bool,
    enable_caching: bool,
) -> bool:
    """Pure policy resolution following capability-based design.

    Returns True if cache creation should proceed, False to skip.
    Consolidates all policy checks in one place for clarity.
    """
    if not enable_caching:
        return False

    # Resolve policy values with defaults
    first_turn_only = (
        policy_hint.first_turn_only
        if policy_hint and policy_hint.first_turn_only is not None
        else True  # Conservative default
    )
    respect_floor = (
        policy_hint.respect_floor
        if policy_hint and policy_hint.respect_floor is not None
        else True  # Conservative default
    )
    conf_skip_floor = (
        policy_hint.conf_skip_floor
        if policy_hint and policy_hint.conf_skip_floor is not None
        else 0.85  # Conservative default
    )

    # First turn policy check
    if first_turn_only and not first_turn:
        return False

    # Token floor policy check
    if respect_floor and token_estimate is not None:
        floor = _resolve_token_floor(model_name, policy_hint)
        if (
            token_estimate.max_tokens < floor
            and token_estimate.confidence >= conf_skip_floor
        ):
            return False

    return True


def _resolve_token_floor(model_name: str, policy_hint: CachePolicyHint | None) -> int:
    """Resolve token floor from policy hint or model capabilities."""
    if policy_hint and policy_hint.min_tokens_floor is not None:
        return int(policy_hint.min_tokens_floor)

    capabilities = get_model_capabilities(model_name)
    if capabilities and capabilities.caching:
        if capabilities.caching.explicit_minimum_tokens:
            return int(capabilities.caching.explicit_minimum_tokens)
        if capabilities.caching.implicit_minimum_tokens:
            return int(capabilities.caching.implicit_minimum_tokens)

    return 4096  # Default fallback


def _is_first_turn(initial: Any) -> bool:
    try:
        turns = tuple(getattr(initial, "history", ()) or ())
        return len(turns) == 0
    except Exception:
        return True


def _select_caching_adapter(
    cfg: Any, adapter_factory: Callable[[str], GenerationAdapter] | None
) -> CachingCapability | None:
    """Return a caching-capable adapter when real API is enabled, else None.

    Keeps the CacheStage provider-neutral and avoids local mock duplication.
    """
    try:
        if adapter_factory is None:
            return None
        api_key = getattr(cfg, "api_key", None)
        if not api_key:
            return None
        adapter = adapter_factory(str(api_key))
        return adapter if isinstance(adapter, CachingCapability) else None
    except Exception:
        return None
