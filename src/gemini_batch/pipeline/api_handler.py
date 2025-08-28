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
from enum import Enum
import logging
import os
from typing import TYPE_CHECKING, Any, TypedDict, cast

# Removed ConfigCompatibilityShim import - no longer needed
from gemini_batch.core.exceptions import APIError
from gemini_batch.core.types import (
    APICall,
    APIPart,
    ExecutionPlan,
    Failure,
    FilePlaceholder,
    FinalizedCommand,
    PlannedCommand,
    Result,
    Success,
    TextPart,
    UploadTask,
)
from gemini_batch.pipeline.adapters.base import (
    ExecutionHintsAware,
    GenerationAdapter,
    UploadsCapability,
)
from gemini_batch.pipeline.base import BaseAsyncHandler
from gemini_batch.pipeline.cache_identity import det_shared_key
from gemini_batch.pipeline.execution_state import ExecutionHints
from gemini_batch.pipeline.hints import CacheHint, ExecutionCacheName
from gemini_batch.telemetry import TelemetryContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemini_batch.pipeline.registries import SimpleRegistry
    from gemini_batch.telemetry import TelemetryContextProtocol

# Note: Provider adapters are injected explicitly via `adapter` or `adapter_factory`.
# No import-time aliasing is required.

logger = logging.getLogger(__name__)


class UploadPhase(str, Enum):
    """Upload processing phases with clear semantic meaning."""

    PARTITION = "partition"  # Separate registry hits from pending uploads
    UPLOAD = "upload"  # Perform concurrent uploads
    REPLACE = "replace"  # Substitute parts with uploaded references

    @classmethod
    def get_telemetry_scope(cls, phase: UploadPhase) -> str:
        """Get standardized telemetry scope name for phase."""
        return f"uploads.{phase.value}"


# --- Telemetry scopes/keys (centralized to avoid typos) ---
T_API_GENERATE = "api.generate"
T_API_HINTS = "api.hints"
T_API_RETRY_NO_CACHE = "api.retry_no_cache"
T_API_GENERATE_RETRY = "api.generate_retry"
T_API_GENERATE_RETRY_LOOP = "api.generate_retry_loop"


class TelemetryUsage(TypedDict, total=False):
    """TypedDict for telemetry usage data structure."""

    total_token_count: int


class TelemetryMetrics(TypedDict, total=False):
    """TypedDict for telemetry metrics data structure."""

    per_prompt: tuple[dict[str, Any], ...]
    vectorized_n_calls: int
    per_call_meta: tuple[dict[str, Any], ...]


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
            # Validate that all calls have non-empty api_parts
            for call in plan.calls:
                if not call.api_parts:
                    raise APIError("API call must have at least one part")

            adapter = self._select_adapter(command)
            # Prepare shared parts once (uploads/registries)
            effective_shared = await self._prepare_shared_parts(adapter, plan)
            finalized = await self._execute_vectorized_calls(
                adapter, command, plan, effective_shared
            )
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
        base_parts: list[APIPart],
        *,
        upload_tasks: tuple[UploadTask, ...] | None = None,
        infer_placeholders: bool = True,
    ) -> list[APIPart]:
        """Sanitize parts and perform upload substitution when supported.

        Callers pass explicit `upload_tasks` from the plan for per-call parts,
        or an empty tuple for shared parts. When `infer_placeholders` is True,
        FilePlaceholder entries are converted into inferred UploadTask entries.
        """
        effective_parts = self._sanitize_history_parts(list(base_parts))

        # Determine upload tasks: prefer explicit, else infer from placeholders
        tasks: tuple[UploadTask, ...] = upload_tasks or ()
        if not tasks and infer_placeholders:
            tasks = self._infer_upload_tasks(effective_parts)

        # If nothing to upload or adapter lacks capability, return early
        if not tasks:
            return effective_parts
        if not isinstance(adapter, UploadsCapability):
            if any(t.required for t in tasks):
                raise APIError("Uploads required but not supported by provider")
            return effective_parts

        # Phase 1: registry reuse vs pending uploads
        with self._telemetry(UploadPhase.get_telemetry_scope(UploadPhase.PARTITION)):
            to_replace, pending = self._partition_uploads(tasks, effective_parts)

        # Phase 2: perform uploads concurrently and update registry
        if pending:
            with self._telemetry(UploadPhase.get_telemetry_scope(UploadPhase.UPLOAD)):
                uploaded_results = await self._upload_pending(adapter, pending)
                to_replace.extend(uploaded_results)

        # Phase 3: coerce to FileRefPart where needed and replace in parts
        with self._telemetry(UploadPhase.get_telemetry_scope(UploadPhase.REPLACE)):
            return self._replace_parts(effective_parts, to_replace)

    # ---- Focused helpers ----
    def _sanitize_history_parts(self, parts: list[APIPart]) -> list[APIPart]:
        from gemini_batch.core.types import HistoryPart

        return [p for p in parts if not (isinstance(p, HistoryPart) and not p.turns)]

    def _infer_upload_tasks(self, parts: list[APIPart]) -> tuple[UploadTask, ...]:
        # Infer from FilePlaceholder instances in parts
        inferred: list[UploadTask] = []
        for idx, p in enumerate(parts):
            if isinstance(p, FilePlaceholder):
                inferred.append(
                    UploadTask(
                        part_index=idx,
                        local_path=p.local_path,
                        mime_type=p.mime_type,
                        required=False,
                    )
                )
        return tuple(inferred)

    def _partition_uploads(
        self, plan_uploads: tuple[UploadTask, ...], parts: list[APIPart]
    ) -> tuple[list[tuple[int, Any]], list[tuple[int, UploadTask]]]:
        to_replace: list[tuple[int, Any]] = []
        pending: list[tuple[int, UploadTask]] = []
        for task in plan_uploads:
            idx = task.part_index
            if idx >= len(parts):
                if task.required:
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
        return to_replace, pending

    async def _upload_pending(
        self, adapter: UploadsCapability, pending: list[tuple[int, UploadTask]]
    ) -> list[tuple[int, Any]]:
        async def _upload_one(i: int, t: UploadTask) -> tuple[int, Any]:
            result = await adapter.upload_file_local(t.local_path, t.mime_type)
            if self._file_registry is not None:
                from contextlib import suppress

                with suppress(Exception):
                    self._file_registry.set(os.fspath(t.local_path), result)
            return i, result

        return await asyncio.gather(*(_upload_one(i, t) for i, t in pending))

    def _coerce_to_file_ref(self, uploaded: Any) -> Any:
        from gemini_batch.core.types import FileRefPart

        if isinstance(uploaded, FileRefPart):
            return uploaded
        # Try attribute-based coercion
        try:
            uri_attr = cast("Any", uploaded).uri
            if isinstance(uri_attr, str):
                return FileRefPart(
                    uri=uri_attr,
                    mime_type=getattr(uploaded, "mime_type", None),
                    raw_provider_data=uploaded,
                )
        except AttributeError:
            pass
        # Try mapping-based coercion
        if (
            isinstance(uploaded, dict)
            and "uri" in uploaded
            and isinstance(uploaded.get("uri"), str)
        ):
            return FileRefPart(
                uri=uploaded["uri"],
                mime_type=cast("Any", uploaded).get("mime_type"),
                raw_provider_data=uploaded,
            )
        return uploaded

    def _replace_parts(
        self, parts: list[APIPart], replacements: list[tuple[int, Any]]
    ) -> list[APIPart]:
        effective = list(parts)
        for idx, uploaded in replacements:
            effective[idx] = self._coerce_to_file_ref(uploaded)
        return effective

    async def _prepare_shared_parts(
        self, adapter: GenerationAdapter, plan: ExecutionPlan
    ) -> list[APIPart]:
        """Prepare effective shared parts once (uploads/registries).

        UploadTasks are not applied to shared parts (indices are per-call), but
        placeholder inference is allowed.
        """
        shared = list(plan.shared_parts)
        return await self._prepare_effective_parts(
            adapter,
            shared,
            upload_tasks=(),
            infer_placeholders=True,
        )

    def _combine_shared_with_call(
        self, shared: list[APIPart], call_parts: tuple[APIPart, ...]
    ) -> tuple[APIPart, ...]:
        """Combine effective shared parts with per-call parts for execution."""
        return tuple(shared) + tuple(call_parts)

    async def _execute_single_call(
        self, adapter: GenerationAdapter, command: PlannedCommand, plan: ExecutionPlan
    ) -> FinalizedCommand:
        """Deprecated path; single-call executes through vectorized machinery."""
        shared = await self._prepare_shared_parts(adapter, plan)
        return await self._execute_vectorized_calls(adapter, command, plan, shared)

    async def _execute_vectorized_calls(
        self,
        adapter: GenerationAdapter,
        command: PlannedCommand,
        plan: ExecutionPlan,
        effective_shared: list[APIPart],
    ) -> FinalizedCommand:
        """Execute vectorized calls with shared context and aggregate telemetry."""
        raw_list: list[dict[str, Any]] = []
        per_prompt_usage: list[dict[str, Any]] = []
        per_call_meta: list[dict[str, Any]] = []

        for call in plan.calls:
            combined_parts = self._combine_shared_with_call(
                effective_shared, call.api_parts
            )
            if isinstance(adapter, _MockAdapter):
                # Deterministic mock echo with per-prompt usage
                ptxt = self._extract_text_from_parts(combined_parts)
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
                    call.cache_name_to_use,
                    had_explicit_cache_plan=bool(call.cache_name_to_use),
                )
            raw_list.append(raw)
            per_prompt_usage.append(dict(cast("dict[str, Any]", raw.get("usage", {}))))
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
        # Attach usage and per-prompt metrics prior to validation
        self._attach_vectorized_usage(
            finalized,
            per_prompt_usage=per_prompt_usage,
            n_calls=len(plan.calls),
            per_call_meta=per_call_meta,
        )

        # Token validation compares estimated aggregate to actual aggregate
        self._attach_token_validation(finalized)
        return finalized

    def _extract_text_from_parts(self, parts: tuple[APIPart, ...]) -> str:
        """Return the last text from parts when present, else empty string.

        This mirrors the behavior used for mock vectorized responses and avoids
        leaking provider-specific shapes.
        """
        for part in reversed(parts):
            if isinstance(part, TextPart):
                return part.text
        return ""

    def _attach_vectorized_usage(
        self,
        finalized: FinalizedCommand,
        *,
        per_prompt_usage: list[dict[str, Any]],
        n_calls: int,
        per_call_meta: list[dict[str, Any]],
    ) -> None:
        """Aggregate and attach vectorized usage and metrics to telemetry."""
        total_tokens = self._sum_usage_total_tokens(per_prompt_usage)
        usage = cast("TelemetryUsage", finalized.telemetry_data.setdefault("usage", {}))
        usage["total_token_count"] = total_tokens
        metrics = cast(
            "TelemetryMetrics", finalized.telemetry_data.setdefault("metrics", {})
        )
        metrics["per_prompt"] = tuple(per_prompt_usage)
        metrics["vectorized_n_calls"] = n_calls
        metrics["per_call_meta"] = tuple(per_call_meta)

    def _sum_usage_total_tokens(self, usage_list: list[dict[str, Any]]) -> int:
        total = 0
        for u in usage_list:
            try:
                total += int(u.get("total_token_count", 0) or 0)
            except Exception:
                total += 0
        return total

    # Cache resolution is handled by CacheStage; no planning-time cache logic here

    async def _execute_with_resilience(
        self,
        adapter: GenerationAdapter,
        command: PlannedCommand,
        primary: APICall,
        parts: tuple[APIPart, ...],
        cache_name: str | None,
        *,
        had_explicit_cache_plan: bool,
    ) -> tuple[dict[str, Any], bool, bool, str | None]:
        used_fallback = False
        retried_without_cache = False
        primary_error_repr: str | None = None
        with self._telemetry("api.execute", model=primary.model_name):
            try:
                # Convert Mapping to dict for the method that needs to modify it
                api_config_dict: dict[str, object] = dict(primary.api_config)
                (
                    raw_response,
                    retried_without_cache,
                ) = await self._generate_with_resilience(
                    adapter,
                    primary.model_name,
                    parts,
                    api_config_dict,
                    cache_name,
                    had_explicit_cache_plan=had_explicit_cache_plan,
                    planned_command=command,
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
                            # Apply explicit no-cache hint for fallback path for uniformity
                            self._apply_adapter_hints(adapter, None)
                            raw_response = await adapter.generate(
                                model_name=fb.model_name,
                                api_parts=tuple(fb.api_parts),
                                api_config=dict(fb.api_config),
                            )
                    # Fallback path does not imply primary no-cache retry
                    retried_without_cache = False
                except Exception as fallback_error:
                    raise APIError(
                        f"Fallback failed after primary error: {primary_error}; fallback error: {fallback_error}"
                    ) from fallback_error
        return raw_response, used_fallback, retried_without_cache, primary_error_repr

    def _build_mock_response(self, command: PlannedCommand) -> dict[str, Any]:
        call0 = command.execution_plan.calls[0]
        return self._build_mock_response_from_parts(tuple(call0.api_parts), command)

    def _build_mock_response_from_parts(
        self, parts: tuple[APIPart, ...], command: PlannedCommand
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
            "model": command.execution_plan.calls[0].model_name,
            "text": f"echo: {first_text}",
            "usage": {
                "prompt_token_count": prompt_tokens,
                "source_token_count": source_tokens,
                "total_token_count": total_tokens,
            },
        }

    def _rebuild_for_fallback(self, command: PlannedCommand) -> PlannedCommand:
        """Return a PlannedCommand whose `calls` contain the fallback APICall.

        Preserves immutability and carries over shared parts, rate constraints,
        and upload tasks to maintain execution semantics in the fallback path.
        """
        plan = command.execution_plan
        if plan.fallback_call is None:
            return command
        from gemini_batch.core.types import ExecutionPlan as _Plan  # local import

        new_plan = _Plan(
            calls=(plan.fallback_call,),
            fallback_call=None,
            shared_parts=plan.shared_parts,
            rate_constraint=plan.rate_constraint,
            upload_tasks=plan.upload_tasks,
        )
        return PlannedCommand(
            resolved=command.resolved,
            execution_plan=new_plan,
            token_estimate=command.token_estimate,
        )

    def _attach_token_validation(self, finalized: FinalizedCommand) -> None:
        # FinalizedCommand guarantees telemetry_data is a dict, TokenEstimate validates its invariants
        estimate = finalized.planned.token_estimate
        if not estimate:
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
    def _read_execution_cache_override(
        self, planned_command: PlannedCommand
    ) -> tuple[str | None, bool]:
        """Return (cache_name_override, overridden) from execution hints.

        Best-effort: never raises. Emits a telemetry gauge in caller.
        """
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
                return exec_cache, True
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(
                "Failed to read execution cache name from hints: %s", e, exc_info=True
            )
        return None, False

    def _apply_adapter_hints(
        self, adapter: GenerationAdapter, cached_content: str | None
    ) -> None:
        """Apply execution hints to adapter when supported (best-effort)."""
        if isinstance(adapter, ExecutionHintsAware):
            from contextlib import suppress

            with suppress(Exception):
                adapter.apply_hints(ExecutionHints(cached_content=cached_content))

    async def _attempt_generate(
        self,
        adapter: GenerationAdapter,
        model_name: str,
        parts: tuple[APIPart, ...],
        api_config: dict[str, object],
        cache_name: str | None,
    ) -> dict[str, Any]:
        """Single generation attempt with optional cache application."""
        self._apply_adapter_hints(adapter, cache_name)
        return await adapter.generate(
            model_name=model_name,
            api_parts=parts,
            api_config=self._with_cache(api_config, cache_name),
        )

    async def _retry_without_cache(
        self,
        adapter: GenerationAdapter,
        model_name: str,
        parts: tuple[APIPart, ...],
        api_config: dict[str, object],
    ) -> dict[str, Any]:
        """One retry without cache, marking telemetry flag for caller."""
        self._apply_adapter_hints(adapter, None)
        return await adapter.generate(
            model_name=model_name,
            api_parts=parts,
            api_config=dict(api_config),
        )

    async def _backoff_generate(
        self,
        adapter: GenerationAdapter,
        model_name: str,
        parts: tuple[APIPart, ...],
        api_config: dict[str, object],
        *,
        attempts: int = 2,
        base_delay: float = 0.5,
        initial_error: Exception,
    ) -> dict[str, Any]:
        """Small backoff loop for transient errors; raises last error if exhausted."""
        # If the initial error is not transient, don't retry pointlessly
        if not self._is_transient_error(initial_error):
            raise initial_error

        last_error: Exception = initial_error
        for i in range(attempts):
            from random import random

            sleep_for = base_delay * (2**i) * (1 + 0.25 * random())  # noqa: S311
            await asyncio.sleep(sleep_for)
            try:
                with self._telemetry(
                    T_API_GENERATE_RETRY, model=model_name, attempt=i + 1
                ):
                    return await adapter.generate(
                        model_name=model_name,
                        api_parts=parts,
                        api_config=dict(api_config),
                    )
            except Exception as e:  # keep last error
                # If the new error is not transient, raise immediately
                if not self._is_transient_error(e):
                    raise e
                last_error = e
        # Exhausted retries; raise the last transient error
        raise last_error

    async def _generate_with_resilience(
        self,
        adapter: GenerationAdapter,
        model_name: str,
        parts: tuple[APIPart, ...],
        api_config: dict[str, object],
        cache_name: str | None,
        *,
        had_explicit_cache_plan: bool,
        planned_command: PlannedCommand,
    ) -> tuple[dict[str, Any], bool]:
        """Generate with cache hinting and minimal retry logic.

        Behavior:
        - If adapter is the mock, execute once deterministically (no retries).
        - If a cache name is present (from plan or exec-time hint), apply it
          and attempt generation.
        - On error and when caching was intended (explicit plan or exec-time
          override), retry once without cache and mark the retry in telemetry.
        - For recognized transient errors, perform a small backoff loop (2 tries).
        """
        retried_without_cache = False

        # Mock path remains deterministic and never retries
        if isinstance(adapter, _MockAdapter):
            with self._telemetry(T_API_GENERATE, model=model_name):
                return self._build_mock_response_from_parts(
                    parts, planned_command
                ), retried_without_cache

        # Read any execution-time cache override from hints (fail-soft)
        cache_overridden_by_hint = False
        override, overridden = self._read_execution_cache_override(planned_command)
        if overridden:
            cache_name = override
            cache_overridden_by_hint = True
            with self._telemetry(T_API_HINTS) as tele:
                tele.gauge("exec_cache_override", 1)

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

            with self._telemetry(T_API_GENERATE, model=model_name) as tele:
                # Lightweight, explicit gauges for clarity
                tele.gauge("cache_intent_plan", 1 if intent_from_plan else 0)
                tele.gauge("cache_intent_override", 1 if intent_from_override else 0)
                tele.gauge("cache_applied", 1 if cache_applied else 0)
                # If cache is applied, write metadata to registry (best-effort)
                if cache_applied:
                    self._write_cache_metadata(
                        planned_command,
                        cast("str", cache_name),
                        applied_via=("override" if intent_from_override else "plan"),
                    )
                raw = await self._attempt_generate(
                    adapter, model_name, parts, api_config, cache_name
                )
                return raw, retried_without_cache
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
                with self._telemetry(T_API_RETRY_NO_CACHE, model=model_name):
                    try:
                        raw = await self._retry_without_cache(
                            adapter, model_name, parts, api_config
                        )
                        retried_without_cache = True
                        return raw, retried_without_cache
                    except Exception:
                        # fall through to backoff retries on transient errors
                        last_error = first_error
            else:
                last_error = first_error

        # Backoff transient errors (raises when exhausted)
        with self._telemetry(T_API_GENERATE_RETRY_LOOP, model=model_name):
            # Defensive: ensure non-None for typing without using assert.
            initial_err: Exception = (
                last_error
                if last_error is not None
                else Exception("initial error missing")
            )
            raw = await self._backoff_generate(
                adapter,
                model_name,
                parts,
                api_config,
                attempts=2,
                base_delay=0.5,
                initial_error=initial_err,
            )
            return raw, retried_without_cache

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

    # --- Cache metadata helpers ---
    def _write_cache_metadata(
        self, planned_command: PlannedCommand, cache_name: str, *, applied_via: str
    ) -> None:
        """Best-effort write of cache metadata to the cache registry.

        Stores the applied cache name and any artifacts from CacheHint under the
        deterministic shared-context key used by CacheStage. Unknown registry
        types are ignored safely.
        """
        reg = self._cache_registry
        if reg is None:
            return
        try:
            initial = planned_command.resolved.initial
            hints = tuple(getattr(initial, "hints", ()) or ())
            cache_hint = next((h for h in hints if isinstance(h, CacheHint)), None)

            plan = planned_command.execution_plan
            model_name = plan.calls[0].model_name
            sys_instr = plan.calls[0].api_config.get("system_instruction")
            sys_text = str(sys_instr) if sys_instr is not None else None

            key = det_shared_key(model_name, sys_text, planned_command)
            setter = getattr(reg, "set_meta", None)
            if callable(setter):
                setter(
                    key,
                    {
                        "cache_name": cache_name,
                        "artifacts": tuple(cache_hint.artifacts) if cache_hint else (),
                        "applied_via": "override"
                        if applied_via == "override"
                        else "plan",
                    },
                )
        except Exception:
            # Best-effort: never fail execution due to metadata
            return


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
