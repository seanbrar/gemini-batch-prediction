"""API handling stage of the pipeline.

Implements a simple, capability-aligned execution flow with:

- Mock-by-default deterministic behavior for tests/examples
- Explicit injection for real provider adapters via constructor
- Upload substitution when supported (with optional task inference)
- Explicit cache creation/use when supported (registry-aware)
- Single fallback attempt when primary execution fails
- Orthogonal telemetry scopes for execute/generate/retry/fallback

Design focuses on data-centricity and simplicity. Neutral types for files and
explicit cache plans live in ``core.types``; this handler consumes them without
leaking provider SDK details.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, cast

# Removed ConfigCompatibilityShim import - no longer needed
from gemini_batch.core.exceptions import APIError
from gemini_batch.core.types import (
    APICall,
    ExecutionPlan,
    Failure,
    FilePlaceholder,
    FinalizedCommand,
    PlannedCommand,
    Result,
    Success,
)
from gemini_batch.pipeline.adapters.base import (
    CachingCapability,
    ExecutionHintsAware,
    GenerationAdapter,
    UploadsCapability,
)
from gemini_batch.pipeline.base import BaseAsyncHandler
from gemini_batch.pipeline.execution_state import ExecutionHints
from gemini_batch.pipeline.hints import CacheHint, ExecutionCacheName
from gemini_batch.telemetry import TelemetryContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemini_batch.core.types import ExplicitCachePlan
    from gemini_batch.pipeline.registries import SimpleRegistry
    from gemini_batch.telemetry import TelemetryContextProtocol

# Note: Provider adapters are injected explicitly via `adapter` or `adapter_factory`.
# No import-time aliasing is required.

logger = logging.getLogger(__name__)


class APIHandler(BaseAsyncHandler[PlannedCommand, FinalizedCommand, APIError]):
    """Executes API calls according to the execution plan.

    Defaults to a deterministic mock. A real Google SDK path is available when
    explicitly enabled via environment or configuration.
    """

    def __init__(
        self,
        telemetry: TelemetryContextProtocol | None = None,
        registries: dict[str, SimpleRegistry] | None = None,
        adapter: GenerationAdapter | None = None,
        adapter_factory: Callable[[str], GenerationAdapter] | None = None,
    ) -> None:
        """Initialize a thin API execution handler with optional telemetry.

        registries: optional mapping with keys "cache" and "files" holding
        CacheRegistry and FileRegistry instances.
        """
        self._telemetry: TelemetryContextProtocol = telemetry or TelemetryContext()
        regs = registries or {}
        self._cache_registry = regs.get("cache")
        self._file_registry = regs.get("files")
        self._adapter: GenerationAdapter | None = adapter
        self._adapter_factory = adapter_factory

    async def handle(
        self, command: PlannedCommand
    ) -> Result[FinalizedCommand, APIError]:
        """Handle the planned command and return a finalized command."""
        try:
            plan = command.execution_plan
            reg_key: str | None = None
            # Vectorized execution path: iterate calls with shared context
            if getattr(plan, "calls", ()):  # non-empty tuple
                adapter = self._select_adapter(command)

                # Prepare shared parts once (uploads/registries)
                effective_shared = await self._prepare_effective_parts(
                    adapter, plan, list(getattr(plan, "shared_parts", ()))
                )

                # Resolve/create shared cache once if supported
                cache_name_ctx: str | None = None
                exp_shared = getattr(plan, "shared_cache", None)
                if exp_shared is not None and isinstance(adapter, CachingCapability):
                    # Registry lookup by deterministic key
                    reg_key = getattr(exp_shared, "deterministic_key", None)
                    if reg_key and self._cache_registry is not None:
                        try:
                            cached_name = self._cache_registry.get(reg_key)
                        except Exception:
                            cached_name = None
                        if isinstance(cached_name, str):
                            cache_name_ctx = cached_name
                    # Create if allowed and not found
                    if not cache_name_ctx and getattr(exp_shared, "create", False):
                        sys_inst = None
                        try:
                            first_call = plan.calls[0]
                            sys_inst = cast(
                                "str | None",
                                first_call.api_config.get("system_instruction"),
                            )
                        except Exception:
                            sys_inst = None
                        created = await adapter.create_cache(
                            model_name=plan.calls[0].model_name,
                            content_parts=tuple(effective_shared),
                            system_instruction=sys_inst,
                            ttl_seconds=getattr(exp_shared, "ttl_seconds", None),
                        )
                        if isinstance(created, str):
                            cache_name_ctx = created
                            if reg_key and self._cache_registry is not None:
                                from contextlib import suppress

                                with suppress(Exception):
                                    self._cache_registry.set(reg_key, created)

                raw_list: list[dict[str, Any]] = []
                per_prompt_usage: list[dict[str, Any]] = []
                per_call_meta: list[dict[str, Any]] = []
                for call in plan.calls:
                    combined_parts = tuple(effective_shared) + tuple(call.api_parts)
                    if isinstance(adapter, _MockAdapter):
                        # Deterministic mock echo with per-prompt usage
                        ptxt = ""
                        for part in reversed(combined_parts):
                            if hasattr(part, "text"):
                                ptxt = cast("Any", part).text
                                break
                        raw = {
                            "mock": True,
                            "model": call.model_name,
                            "text": f"echo: {ptxt}",
                            "usage": {
                                "prompt_token_count": max(len(ptxt) // 4 + 10, 0),
                                "source_token_count": 0,
                                "total_token_count": max(len(ptxt) // 4 + 10, 0),
                            },
                        }
                        used_fallback = False
                        retried_without_cache = False
                        primary_error_repr = None
                    else:
                        # Use the same resilience path as single-call execution
                        (
                            raw,
                            used_fallback,
                            retried_without_cache,
                            primary_error_repr,
                        ) = await self._execute_with_resilience(
                            adapter,
                            command,
                            call,
                            tuple(combined_parts),
                            cache_name_ctx,
                            had_explicit_cache_plan=(exp_shared is not None),
                        )
                    raw_list.append(raw)
                    per_prompt_usage.append(
                        dict(cast("dict[str, Any]", raw.get("usage", {})))
                    )
                    meta: dict[str, Any] = {}
                    if used_fallback:
                        meta["used_fallback"] = True
                    if retried_without_cache:
                        meta["retried_without_cache"] = True
                    if primary_error_repr:
                        meta["primary_error"] = primary_error_repr
                    per_call_meta.append(meta)

                # Build finalized result with aggregated usage in telemetry
                finalized = FinalizedCommand(
                    planned=command,
                    raw_api_response={
                        "model": plan.calls[0].model_name,
                        "batch": tuple(raw_list),
                    },
                )
                # Aggregate total tokens for validation/metrics
                total_tokens = 0
                for u in per_prompt_usage:
                    try:
                        total_tokens += int(u.get("total_token_count", 0) or 0)
                    except Exception:
                        total_tokens += 0

                # Attach usage and per-prompt metrics prior to validation
                finalized.telemetry_data.setdefault("usage", {})
                cast("dict[str, Any]", finalized.telemetry_data["usage"]).update(
                    {"total_token_count": total_tokens}
                )
                cast(
                    "dict[str, Any]", finalized.telemetry_data.setdefault("metrics", {})
                ).update(
                    {
                        "per_prompt": tuple(per_prompt_usage),
                        "vectorized_n_calls": len(plan.calls),
                        "per_call_meta": tuple(per_call_meta),
                    }
                )

                # Token validation compares estimated aggregate to actual aggregate
                self._attach_token_validation(finalized)
                return Success(finalized)

            primary = plan.primary_call
            if not primary.api_parts:
                return Failure(APIError("Execution plan has no parts to execute"))

            # 1) Select provider adapter
            adapter = self._select_adapter(command)

            # 2) Prepare effective parts (uploads, inference, substitution)
            effective_parts = await self._prepare_effective_parts(
                adapter, plan, list(primary.api_parts)
            )

            # 3) Resolve cache name (explicit or hint)
            cache_name, exp_cache = await self._resolve_cache(
                adapter, plan, effective_parts, primary
            )

            # Best-effort: persist cache artifacts metadata using the registry's
            # metadata channel when available (keeps primary values homogeneous).
            try:
                reg = self._cache_registry
                if reg is not None and cache_name:
                    # Read artifacts from the first CacheHint on the initial command
                    initial = command.resolved.initial
                    ch = next(
                        (
                            h
                            for h in (getattr(initial, "hints", ()) or ())
                            if isinstance(h, CacheHint)
                        ),
                        None,
                    )
                    reg_key = getattr(
                        getattr(plan, "explicit_cache", None), "deterministic_key", None
                    )
                    if ch is not None and reg_key:
                        try:
                            from gemini_batch.pipeline.registries import CacheRegistry

                            if isinstance(reg, CacheRegistry):
                                reg.set_meta(
                                    reg_key,
                                    {
                                        "cache_name": cache_name,
                                        "artifacts": tuple(ch.artifacts),
                                    },
                                )
                        except Exception as inner:
                            logger.debug(
                                "Cache metadata set_meta failed (key=%s): %s",
                                reg_key,
                                inner,
                                exc_info=True,
                            )
            except Exception as e:
                logger.debug(
                    "Cache metadata write skipped (key=%s): %s",
                    reg_key,
                    e,
                    exc_info=True,
                )
                # Strictly best-effort; never fail execution due to metadata write

            # 4) Execute with resilience and optional fallback
            (
                raw_response,
                used_fallback,
                retried_without_cache,
                primary_error_repr,
            ) = await self._execute_with_resilience(
                adapter,
                command,
                primary,
                tuple(effective_parts),
                cache_name,
                had_explicit_cache_plan=(exp_cache is not None),
            )

            finalized = FinalizedCommand(planned=command, raw_api_response=raw_response)
            self._attach_token_validation(finalized)
            self._attach_usage_data(finalized)
            # Best-effort attachment of execution metadata
            exec_meta = cast(
                "dict[str, object]",
                finalized.telemetry_data.setdefault("execution", {}),
            )
            if retried_without_cache:
                exec_meta["retried_without_cache"] = True
            if used_fallback:
                exec_meta["used_fallback"] = True
            if primary_error_repr:
                exec_meta.setdefault("primary_error", primary_error_repr)
            return Success(finalized)
        except APIError as e:
            return Failure(e)
        except Exception as e:  # Defensive normalization
            return Failure(APIError(f"API handler failed: {e}"))

    # --- Internal helpers ---

    def _select_adapter(self, command: PlannedCommand) -> GenerationAdapter:
        if self._adapter is not None:
            return self._adapter
        if self._adapter_factory is not None:
            config = command.resolved.initial.config
            api_key = config.api_key
            if not api_key:
                raise APIError("Adapter factory provided but api_key missing")
            try:
                return self._adapter_factory(str(api_key))
            except Exception as e:  # pragma: no cover
                raise APIError(f"Failed to initialize provider: {e}") from e
        return _MockAdapter()

    async def _prepare_effective_parts(
        self,
        adapter: GenerationAdapter,
        plan: ExecutionPlan,
        base_parts: list[Any],
    ) -> list[Any]:
        effective_parts: list[Any] = list(base_parts)
        # A2 shim: render ConversationHistoryPart into a TextPart for adapters
        from gemini_batch.core.types import ConversationHistoryPart, TextPart

        def _render_history(turns: tuple[Any, ...]) -> str:
            if not turns:
                return ""
            lines: list[str] = []
            for t in turns:
                u = getattr(t, "question", None)
                a = getattr(t, "answer", None)
                if u:
                    lines.append(f"User: {u}")
                if a:
                    lines.append(f"Assistant: {a}")
            return "\n".join(lines)

        # Create a copy for safe in-place edits and index management
        idx = 0
        while idx < len(effective_parts):
            part = effective_parts[idx]
            if isinstance(part, ConversationHistoryPart):
                rendered = _render_history(part.turns)
                if rendered:
                    effective_parts[idx] = TextPart(text=rendered)
                    idx += 1
                else:
                    # Drop empty history to avoid blank parts
                    effective_parts.pop(idx)
                    # do not increment idx so the next element shifts into current index
            else:
                idx += 1
        plan_uploads = getattr(plan, "upload_tasks", ())
        # Infer uploads from placeholders when not explicitly planned
        if not plan_uploads:
            inferred: list[Any] = []
            for idx, p in enumerate(effective_parts):
                if isinstance(p, FilePlaceholder):
                    inferred.append(
                        type(
                            "_Task",
                            (),
                            {
                                "part_index": idx,
                                "local_path": p.local_path,
                                "mime_type": p.mime_type,
                                "required": False,
                            },
                        )()
                    )
            plan_uploads = tuple(inferred)

        if plan_uploads:
            if not isinstance(adapter, UploadsCapability):
                if any(getattr(t, "required", True) for t in plan_uploads):
                    raise APIError("Uploads required but not supported by provider")
                return effective_parts

            # Phase 1: use registry or schedule uploads
            to_replace: list[tuple[int, Any]] = []
            pending: list[tuple[int, Any]] = []
            for task in plan_uploads:
                idx = int(task.part_index)
                if idx < 0 or idx >= len(effective_parts):
                    if getattr(task, "required", True):
                        raise APIError(f"UploadTask index {idx} out of range")
                    continue
                local_id = os.fspath(task.local_path)
                uploaded: Any | None = None
                if self._file_registry is not None:
                    try:
                        uploaded = self._file_registry.get(local_id)
                    except Exception:
                        uploaded = None
                if uploaded is not None:
                    to_replace.append((idx, uploaded))
                else:
                    pending.append((idx, task))

            async def _upload_one(i: int, t: Any) -> tuple[int, Any]:
                result = await adapter.upload_file_local(
                    t.local_path, getattr(t, "mime_type", None)
                )
                if self._file_registry is not None:
                    from contextlib import suppress

                    with suppress(Exception):
                        self._file_registry.set(os.fspath(t.local_path), result)
                return i, result

            if pending:
                uploaded_results = await asyncio.gather(
                    *(_upload_one(i, t) for i, t in pending)
                )
                to_replace.extend(uploaded_results)

            # Phase 3: coerce and replace
            from gemini_batch.core.types import (
                FileRefPart,  # local import to avoid cycles
            )

            for idx, uploaded in to_replace:
                if isinstance(uploaded, FileRefPart):
                    effective_parts[idx] = uploaded
                    continue
                coerced = None
                try:
                    uri_attr = cast("Any", uploaded).uri
                    if isinstance(uri_attr, str):
                        coerced = FileRefPart(
                            uri=uri_attr, mime_type=getattr(uploaded, "mime_type", None)
                        )
                except Exception:
                    coerced = None
                if coerced is None and isinstance(uploaded, dict) and "uri" in uploaded:
                    u = uploaded.get("uri")
                    if isinstance(u, str):
                        coerced = FileRefPart(
                            uri=u, mime_type=cast("Any", uploaded).get("mime_type")
                        )
                effective_parts[idx] = coerced if coerced is not None else uploaded

        return effective_parts

    async def _resolve_cache(
        self,
        adapter: GenerationAdapter,
        plan: ExecutionPlan,
        effective_parts: list[Any],
        primary: APICall,
    ) -> tuple[str | None, ExplicitCachePlan | None]:
        """Resolve the cache name to use for this execution.

        Precedence (fail-soft):
        1) Start with any `APICall.cache_name_to_use` carried in the plan.
        2) If an `ExplicitCachePlan` is present and the adapter supports caching,
           consult the cache registry by deterministic key and, if missing,
           create the cache from the selected content parts (optionally including
           the system instruction). Store the created name in the registry.
        3) If `ExplicitCachePlan.cache_name` is set but `create=False`, use it.

        Returns a tuple of `(cache_name, explicit_cache_plan)` to inform retry
        semantics upstream; callers may still apply an execution-time override
        via `ExecutionCacheName` later.
        """
        cache_name = primary.cache_name_to_use
        exp_cache = getattr(plan, "explicit_cache", None)
        if exp_cache is not None and isinstance(adapter, CachingCapability):
            if getattr(exp_cache, "create", False):
                content_parts = [
                    effective_parts[i]
                    for i in getattr(exp_cache, "contents_part_indexes", ())
                    if 0 <= i < len(effective_parts)
                ]
                reg_key = getattr(exp_cache, "deterministic_key", None)
                if reg_key and self._cache_registry is not None:
                    try:
                        cached_name = self._cache_registry.get(reg_key)
                    except Exception:
                        cached_name = None
                    if isinstance(cached_name, str):
                        cache_name = cached_name
                if not cache_name:
                    created = await adapter.create_cache(
                        model_name=primary.model_name,
                        content_parts=tuple(content_parts),
                        system_instruction=(
                            cast(
                                "str | None",
                                primary.api_config.get("system_instruction"),
                            )
                            if getattr(exp_cache, "include_system_instruction", True)
                            else None
                        ),
                        ttl_seconds=getattr(exp_cache, "ttl_seconds", None),
                    )
                    if isinstance(created, str):
                        cache_name = created
                        if reg_key and self._cache_registry is not None:
                            from contextlib import suppress

                            with suppress(Exception):
                                self._cache_registry.set(reg_key, created)
            elif getattr(exp_cache, "cache_name", None):
                cache_name = exp_cache.cache_name
        return cache_name, exp_cache

    async def _execute_with_resilience(
        self,
        adapter: GenerationAdapter,
        command: PlannedCommand,
        primary: APICall,
        parts: tuple[Any, ...],
        cache_name: str | None,
        *,
        had_explicit_cache_plan: bool,
    ) -> tuple[dict[str, Any], bool, bool, str | None]:
        used_fallback = False
        retried_without_cache = False
        primary_error_repr: str | None = None
        with self._telemetry("api.execute", model=primary.model_name):
            try:
                raw_response = await self._generate_with_resilience(
                    adapter,
                    primary.model_name,
                    parts,
                    primary.api_config,
                    cache_name,
                    had_explicit_cache_plan=had_explicit_cache_plan,
                    planned_command=command,
                )
                retried_without_cache = getattr(
                    self, "_last_retry_without_cache", False
                )
            except Exception as primary_error:
                if command.execution_plan.fallback_call is None:
                    raise APIError(
                        f"Provider call failed: {primary_error}"
                    ) from primary_error
                try:
                    fb = command.execution_plan.fallback_call
                    if fb is None:
                        raise APIError("Fallback plan unexpectedly missing")
                    with self._telemetry("api.fallback", model=fb.model_name):
                        used_fallback = True
                        primary_error_repr = str(primary_error)
                        if isinstance(adapter, _MockAdapter):
                            raw_response = self._build_mock_response(
                                self._rebuild_for_fallback(command)
                            )
                        else:
                            raw_response = await adapter.generate(
                                model_name=fb.model_name,
                                api_parts=tuple(fb.api_parts),
                                api_config=dict(fb.api_config),
                            )
                except Exception as fallback_error:
                    raise APIError(
                        f"Fallback failed after primary error: {primary_error}; fallback error: {fallback_error}"
                    ) from fallback_error
        return raw_response, used_fallback, retried_without_cache, primary_error_repr

    def _build_mock_response(self, command: PlannedCommand) -> dict[str, Any]:
        primary = command.execution_plan.primary_call
        return self._build_mock_response_from_parts(tuple(primary.api_parts), command)

    def _build_mock_response_from_parts(
        self, parts: tuple[Any, ...], command: PlannedCommand
    ) -> dict[str, Any]:
        first_text = ""
        if parts:
            p0 = parts[0]
            first_text = cast("Any", p0).text if hasattr(p0, "text") else str(p0)
        estimate = command.token_estimate
        if estimate is not None:
            prompt_tokens = max(len(first_text) // 4 + 10, 0)
            source_tokens = max(estimate.expected_tokens - prompt_tokens, 0)
            total_tokens = prompt_tokens + source_tokens
        else:
            prompt_tokens = len(first_text) // 4 + 10
            source_tokens = 0
            total_tokens = prompt_tokens

        return {
            "mock": True,
            "model": command.execution_plan.primary_call.model_name,
            "text": f"echo: {first_text}",
            "usage": {
                "prompt_token_count": prompt_tokens,
                "source_token_count": source_tokens,
                "total_token_count": total_tokens,
            },
        }

    def _rebuild_for_fallback(self, command: PlannedCommand) -> PlannedCommand:
        """Return a PlannedCommand whose primary call is the fallback call.

        This preserves immutability by constructing a shallow replacement shape.
        """
        plan = command.execution_plan
        if plan.fallback_call is None:
            return command
        from gemini_batch.core.types import ExecutionPlan as _Plan  # local import

        new_plan = _Plan(primary_call=plan.fallback_call, fallback_call=None)
        return PlannedCommand(
            resolved=command.resolved,
            execution_plan=new_plan,
            token_estimate=command.token_estimate,
        )

    def _attach_token_validation(self, finalized: FinalizedCommand) -> None:
        estimate = finalized.planned.token_estimate
        if not estimate or not isinstance(finalized.telemetry_data, dict):
            return
        usage: dict[str, Any] = {}
        try:
            raw = finalized.raw_api_response
            if isinstance(raw, dict):
                usage = cast("dict[str, Any]", raw.get("usage", {}))
        except Exception:
            usage = {}
        actual = (
            int(usage.get("total_token_count", 0)) if isinstance(usage, dict) else 0
        )
        # Vectorized path: fall back to telemetry usage totals when raw lacks usage
        if actual == 0:
            try:
                tele_usage = finalized.telemetry_data.get("usage", {})
                if isinstance(tele_usage, dict):
                    actual = int(tele_usage.get("total_token_count", 0) or 0)
            except Exception:
                actual = 0
        cast(
            "dict[str, object]",
            finalized.telemetry_data.setdefault("token_validation", {}),
        ).update(
            {
                "estimated_expected": estimate.expected_tokens,
                "estimated_min": estimate.min_tokens,
                "estimated_max": estimate.max_tokens,
                "actual": actual,
                "in_range": estimate.min_tokens <= actual <= estimate.max_tokens,
            }
        )

    def _attach_usage_data(self, finalized: FinalizedCommand) -> None:
        """Extract usage data from API response and attach to telemetry."""
        if not isinstance(finalized.telemetry_data, dict):
            return
        usage: dict[str, Any] = {}
        try:
            raw = finalized.raw_api_response
            if isinstance(raw, dict):
                usage = cast("dict[str, Any]", raw.get("usage", {}))
        except Exception:
            usage = {}

        if usage:
            finalized.telemetry_data["usage"] = dict(usage)

    def _with_cache(
        self, api_config: dict[str, object], cache_name: str | None
    ) -> dict[str, object]:
        if not cache_name:
            return dict(api_config)
        cfg = dict(api_config)
        # Provider-specific adapters may interpret this key; harmless for mock
        cfg.setdefault("cached_content", cache_name)
        return cfg

    # --- Resilience helpers ---
    async def _generate_with_resilience(
        self,
        adapter: GenerationAdapter,
        model_name: str,
        parts: tuple[Any, ...],
        api_config: dict[str, object],
        cache_name: str | None,
        *,
        had_explicit_cache_plan: bool,
        planned_command: PlannedCommand,
    ) -> dict[str, Any]:
        """Generate with cache hinting and minimal retry logic.

        Behavior:
        - If adapter is the mock, execute once deterministically (no retries).
        - If a cache name is present (from plan or exec-time hint), apply it
          and attempt generation.
        - On error and when caching was intended (explicit plan or exec-time
          override), retry once without cache and mark the retry in telemetry.
        - For recognized transient errors, perform a small backoff loop (2 tries).
        """
        # Reset flag visible to caller for telemetry
        self._last_retry_without_cache = False

        # Mock path remains deterministic and never retries
        if isinstance(adapter, _MockAdapter):
            with self._telemetry("api.generate", model=model_name):
                return self._build_mock_response_from_parts(parts, planned_command)

        # Read any execution-time cache override from hints (fail-soft)
        cache_overridden_by_hint = False
        try:
            initial = planned_command.resolved.initial
            hints_tuple = tuple(getattr(initial, "hints", ()) or ())
            exec_cache = next(
                (
                    h.cache_name
                    for h in hints_tuple
                    if isinstance(h, ExecutionCacheName)
                ),
                None,
            )
            if exec_cache:
                cache_name = exec_cache
                cache_overridden_by_hint = True
                # Emit telemetry for cache override
                with self._telemetry("api.hints") as tele:
                    tele.gauge("exec_cache_override", 1)
        except Exception as e:
            logger.debug(
                "Failed to read execution cache name from hints: %s",
                e,
                exc_info=True,
            )
            # strictly best-effort; never fail on hints

        # Real path: attempt with cache hint first (if provided)
        last_error: Exception | None = None
        try:
            # Clarify intent derivation for observability and correctness.
            # intent_from_plan: planner produced an explicit cache plan
            # intent_from_override: execution-time override via ExecutionCacheName
            # cache_applied: a concrete cache name is present for this attempt
            intent_from_plan = bool(had_explicit_cache_plan)
            intent_from_override = bool(cache_overridden_by_hint)
            cache_applied = cache_name is not None

            hints = ExecutionHints(cached_content=cache_name)
            if isinstance(adapter, ExecutionHintsAware):
                from contextlib import suppress

                with suppress(Exception):
                    adapter.apply_hints(hints)
            with self._telemetry("api.generate", model=model_name) as tele:
                # Lightweight, explicit gauges for clarity
                tele.gauge("cache_intent_plan", 1 if intent_from_plan else 0)
                tele.gauge("cache_intent_override", 1 if intent_from_override else 0)
                tele.gauge("cache_applied", 1 if cache_applied else 0)
                return await adapter.generate(
                    model_name=model_name,
                    api_parts=parts,
                    api_config=self._with_cache(api_config, cache_name),
                )
        except Exception as first_error:
            # Retry w/o cache only if caching was truly intended and applied:
            #   a) explicit cache plan AND a concrete cache name was applied; or
            #   b) an execution-time override provided a cache name.
            intent_from_plan = bool(had_explicit_cache_plan)
            intent_from_override = bool(cache_overridden_by_hint)
            cache_applied = cache_name is not None
            treat_as_explicit = (
                intent_from_plan and cache_applied
            ) or intent_from_override
            if treat_as_explicit:
                with self._telemetry("api.retry_no_cache", model=model_name):
                    self._last_retry_without_cache = True
                    try:
                        hints = ExecutionHints(cached_content=None)
                        if isinstance(adapter, ExecutionHintsAware):
                            from contextlib import suppress

                            with suppress(Exception):
                                adapter.apply_hints(hints)
                        return await adapter.generate(
                            model_name=model_name,
                            api_parts=parts,
                            api_config=dict(api_config),
                        )
                    except Exception:
                        # fall through to backoff retries on transient errors
                        last_error = first_error
            else:
                last_error = first_error

        # Backoff loop for transient errors (2 attempts)
        attempts = 2
        delay = 0.5
        for i in range(attempts):
            from random import random

            if last_error is None or not self._is_transient_error(last_error):
                break
            sleep_for = (
                delay * (2**i) * (1 + 0.25 * random())  # noqa: S311
            )  # Cryptographic weakness is fine
            await asyncio.sleep(sleep_for)
            try:
                with self._telemetry(
                    "api.generate_retry", model=model_name, attempt=i + 1
                ):
                    return await adapter.generate(
                        model_name=model_name,
                        api_parts=parts,
                        api_config=dict(api_config),
                    )
            except Exception as e:  # keep last error
                last_error = e

        # Exhausted retries
        raise last_error

    def _is_transient_error(self, err: Exception) -> bool:
        text = str(err).lower()
        return (
            "timeout" in text
            or "timed out" in text
            or "429" in text
            or "rate limit" in text
            or "temporarily" in text
            or "unavailable" in text
        )


class _MockAdapter(GenerationAdapter):
    """Deterministic adapter used for tests/examples (no network)."""

    async def upload_file_local(
        self, path: os.PathLike[str] | str, mime_type: str | None
    ) -> Any:
        # Return a neutral FileRefPart-like mapping with a fake URI
        from gemini_batch.core.types import FileRefPart

        return FileRefPart(
            uri=f"mock://uploaded/{os.fspath(path)}", mime_type=mime_type
        )

    async def create_cache(
        self,
        *,
        model_name: str,
        content_parts: tuple[Any, ...],
        system_instruction: str | None,
        ttl_seconds: int | None,  # noqa: ARG002
    ) -> str:
        # Deterministic pseudo cache name for testing; model-bound
        base = (system_instruction or "") + "|".join(
            str(getattr(p, "uri", getattr(p, "text", p))) for p in content_parts
        )
        suffix = hex(abs(hash((model_name, base))) % (1 << 32))[2:]
        return f"cachedContents/mock-{model_name}-{suffix}"

    async def generate(
        self,
        *,
        model_name: str,
        api_parts: tuple[Any, ...],
        api_config: dict[str, object],  # noqa: ARG002
    ) -> dict[str, Any]:
        # The handler builds the final mock response to incorporate token estimate logic.
        # This adapter simply echoes the first text part to keep behavior explicit.
        first_text = ""
        try:
            part0 = next(iter(api_parts))
            if hasattr(part0, "text"):
                first_text = cast("Any", part0).text
            elif isinstance(part0, dict) and "text" in part0:
                first_text = str(part0["text"])  # pragma: no cover
            else:
                first_text = str(part0)
        except StopIteration:
            first_text = ""
        return {"model": model_name, "text": f"echo: {first_text}"}


## Note: Real provider adapter implementation lives in
## `gemini_batch.pipeline.adapters.gemini.GoogleGenAIAdapter` and should be
## passed explicitly via `APIHandler(adapter=...)` or `adapter_factory=...`.
