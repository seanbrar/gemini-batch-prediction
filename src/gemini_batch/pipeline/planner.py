"""Execution planning stage of the pipeline.

This module contains the handler responsible for creating execution plans
from resolved commands, determining how to process the request efficiently.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

from gemini_batch.core.exceptions import ConfigurationError
from gemini_batch.core.models import get_model_capabilities
from gemini_batch.core.types import (
    APICall,
    ExecutionPlan,
    ExplicitCachePlan,
    Failure,
    FilePlaceholder,
    PlannedCommand,
    PromptBundle,
    RateConstraint,
    ResolvedCommand,
    Result,
    Success,
    TextPart,
    TokenEstimate,
    UploadTask,
)
from gemini_batch.pipeline.base import BaseAsyncHandler
from gemini_batch.pipeline.hints import CacheHint, EstimationOverrideHint
from gemini_batch.pipeline.prompts import assemble_prompts
from gemini_batch.pipeline.tokens.adapters.gemini import (
    GeminiEstimationAdapter,
)
from gemini_batch.telemetry import TelemetryContext

if TYPE_CHECKING:
    from gemini_batch.telemetry import TelemetryContextProtocol

    from .tokens.adapters.base import EstimationAdapter  # pragma: no cover


class ExecutionPlanner(
    BaseAsyncHandler[ResolvedCommand, PlannedCommand, ConfigurationError]
):
    """Creates execution plans from resolved commands (minimal slice).

    Minimal responsibilities implemented now:
    - Assemble prompt text from the initial command
    - Create a trivial APICall with no caching decisions
    - Package into an ExecutionPlan and return a PlannedCommand

    Future iterations will add token estimation, caching strategies, and
    payload/file decisions per the architecture spec.
    """

    def __init__(
        self,
        estimation_adapter: EstimationAdapter | None = None,
        telemetry: TelemetryContextProtocol | None = None,
    ) -> None:
        """Initialize a planner with optional estimation and telemetry.

        Defaults to a Gemini-specific estimation adapter. Telemetry is optional
        and incurs zero overhead when not provided or disabled.
        """
        # Adapter is library-owned and provider-neutral at this seam.
        # Use a precise type annotation to keep mypy strict without importing
        # the protocol at runtime.
        self._adapter: EstimationAdapter
        if estimation_adapter is not None:
            self._adapter = estimation_adapter
        else:
            # Instantiate the default Gemini estimation adapter eagerly.
            self._adapter = cast("Any", GeminiEstimationAdapter)()
        # Safe no-op context when not enabled
        self._telemetry: TelemetryContextProtocol = telemetry or TelemetryContext()

    async def handle(
        self, command: ResolvedCommand
    ) -> Result[PlannedCommand, ConfigurationError]:
        """Create a minimal execution plan for the resolved command.

        This stage assembles a single `APICall` from the input prompts and
        resolved sources, computes a token estimate, and optionally attaches
        caching and rate constraints.

        Args:
            command: The resolved command (sources are already materialized).

        Returns:
            Success with `PlannedCommand` on success, otherwise a failure with
            `ConfigurationError`.
        """
        try:
            initial = command.initial
            config = initial.config
            model_name: str = str(config.model or "gemini-2.0-flash")

            # --- Hints (kept optional and fail-soft) ---
            hints = tuple(getattr(initial, "hints", ()) or ())
            cacheh = next((h for h in hints if isinstance(h, CacheHint)), None)
            overh = next(
                (h for h in hints if isinstance(h, EstimationOverrideHint)), None
            )

            # Emit minimal hint telemetry
            with self._telemetry("planner.hints") as tele:
                tele.gauge("hints_seen", len(hints))
                if cacheh:
                    tele.gauge("cache_hint", 1)
                if overh:
                    tele.gauge("estimation_override", 1)

            # Assemble prompts using the prompt assembly system
            try:
                prompt_bundle: PromptBundle = assemble_prompts(command)
            except ConfigurationError as e:
                return Failure(e)

            # Join user prompts for token estimation and API call
            joined_prompt = "\n\n".join(prompt_bundle.user)

            # Emit prompt assembly telemetry
            with self._telemetry("planner.prompt") as tele:
                tele.gauge("user_from", prompt_bundle.hints.get("user_from", "unknown"))
                tele.gauge(
                    "system_from", prompt_bundle.hints.get("system_from", "none")
                )
                if prompt_bundle.system:
                    tele.gauge("system_len", len(prompt_bundle.system))
                tele.gauge("user_total_len", sum(len(p) for p in prompt_bundle.user))
                if system_file := prompt_bundle.hints.get("system_file"):
                    tele.gauge("system_file", system_file)
                if user_file := prompt_bundle.hints.get("user_file"):
                    tele.gauge("user_file", user_file)

            # Estimate tokens for prompt and resolved sources (pure, adapter-based)
            with self._telemetry("planner.estimate", model=model_name):
                # Fabricate a lightweight text Source for prompt estimation
                from gemini_batch.core.types import (
                    Source,  # local import to avoid cycles
                )

                prompt_source = Source(
                    source_type="text",
                    identifier=joined_prompt,
                    mime_type="text/plain",
                    size_bytes=len(joined_prompt.encode("utf-8")),
                    content_loader=lambda: joined_prompt.encode("utf-8"),
                )

                source_estimates = [
                    self._adapter.estimate(s) for s in command.resolved_sources
                ]
                prompt_estimate = self._adapter.estimate(prompt_source)
                combined_estimates = [prompt_estimate, *source_estimates]
                aggregated: TokenEstimate = self._adapter.aggregate(combined_estimates)

                # Apply conservative, planner-scoped overrides (no provider coupling)
                aggregated = self._apply_estimation_override(aggregated, overh)

                # Normalize breakdown to include a stable 'prompt' key.
                # Adapters usually return the prompt as the first item; we
                # re-label it to 'prompt' to keep downstream consumers stable.
                breakdown: dict[str, TokenEstimate] | None = None
                if aggregated.breakdown:
                    breakdown = {}
                    for idx, (k, v) in enumerate(aggregated.breakdown.items()):
                        # The adapter labels as source_0, source_1, ... based on input order
                        breakdown["prompt" if idx == 0 else k] = v
                total_estimate = TokenEstimate(
                    min_tokens=aggregated.min_tokens,
                    expected_tokens=aggregated.expected_tokens,
                    max_tokens=aggregated.max_tokens,
                    confidence=aggregated.confidence,
                    breakdown=breakdown,
                )

            # Upload tasks are not planned until a richer parts mapping exists.
            # For now, the API handler infers uploads from placeholders.
            upload_tasks: tuple[UploadTask, ...] = ()

            # Caching decision: conservative based on max_tokens
            explicit_cache: ExplicitCachePlan | None = None
            # Honor either the planner policy OR an explicit CacheHint
            if self._should_cache(total_estimate, config) or cacheh is not None:
                key = self._deterministic_cache_key(
                    command, joined_prompt, prompt_bundle.system
                )
                # In this minimal slice, cache the prompt (part index 0) and include system_instruction when present
                explicit_cache = ExplicitCachePlan(
                    create=not (cacheh and cacheh.reuse_only),
                    cache_name=None,
                    contents_part_indexes=(0,),
                    include_system_instruction=True,
                    ttl_seconds=(
                        cacheh.ttl_seconds if cacheh is not None else config.ttl_seconds
                    ),
                    deterministic_key=(
                        cacheh.deterministic_key if cacheh is not None else key
                    ),
                )
                # For backward compatibility with tests expecting a cache hint
                # provide a deterministic cache_name_to_use hint.
                cache_hint = self._generate_cache_name(command, prompt_bundle.system)
            else:
                cache_hint = None

            # Build parts: prompt first, then map file sources to placeholders (MVP)

            parts: list[Any] = [TextPart(text=joined_prompt)]
            for s in command.resolved_sources:
                if s.source_type == "file":
                    from pathlib import Path

                    parts.append(
                        FilePlaceholder(
                            local_path=Path(str(s.identifier)), mime_type=s.mime_type
                        )
                    )

            # Create API config with system instruction when present
            api_config: dict[str, Any] = {}
            if prompt_bundle.system:
                api_config["system_instruction"] = prompt_bundle.system

            api_call = APICall(
                model_name=model_name,
                api_parts=tuple(parts),
                api_config=api_config,
                cache_name_to_use=cache_hint,
            )

            # Resolve rate limits (vendor-neutral via core.models) only for real API runs.
            # In dry runs (use_real_api=False) do not attach any constraints to avoid
            # artificial delays and to keep handlers context-free. The pipeline always
            # includes the RateLimitHandler; enforcement is controlled solely by the
            # presence (or absence) of this constraint in the plan.
            rate_constraint: RateConstraint | None = None
            if config.use_real_api:
                try:
                    from gemini_batch.core.models import get_rate_limits

                    tier = config.tier
                    limits = (
                        get_rate_limits(tier, model_name) if tier is not None else None
                    )
                    if limits is not None:
                        rate_constraint = RateConstraint(
                            requests_per_minute=limits.requests_per_minute,
                            tokens_per_minute=limits.tokens_per_minute,
                        )
                except Exception:
                    rate_constraint = None

            plan = ExecutionPlan(
                primary_call=api_call,
                fallback_call=None,
                rate_constraint=rate_constraint,
                upload_tasks=upload_tasks,
                explicit_cache=explicit_cache,
            )
            planned = PlannedCommand(
                resolved=command,
                execution_plan=plan,
                token_estimate=total_estimate,
            )
            return Success(planned)
        except ConfigurationError as e:
            return Failure(e)
        except Exception as e:  # Defensive: normalize unexpected errors
            return Failure(ConfigurationError(f"Failed to plan execution: {e}"))

    # --- Internal helpers ---
    def _apply_estimation_override(
        self, estimate: TokenEstimate, override_hint: EstimationOverrideHint | None
    ) -> TokenEstimate:
        """Apply conservative token estimation overrides while maintaining invariants.

        Applies widen-then-clamp logic to max_tokens and ensures expected_tokens
        remains within [min_tokens, max_tokens]. Returns original estimate if no override.
        """
        if override_hint is None:
            return estimate

        # Widen max_tokens by factor
        new_max = estimate.max_tokens
        factor = float(override_hint.widen_max_factor)
        if factor and factor != 1.0:
            new_max = int(new_max * factor)

        # Apply optional upper clamp
        if override_hint.clamp_max_tokens is not None:
            new_max = min(new_max, int(override_hint.clamp_max_tokens))

        # Ensure max >= min (invariant enforcement)
        new_max = max(new_max, estimate.min_tokens)

        # Keep expected within [min, max] bounds
        new_expected = max(estimate.min_tokens, min(estimate.expected_tokens, new_max))

        return TokenEstimate(
            min_tokens=estimate.min_tokens,
            expected_tokens=new_expected,
            max_tokens=new_max,
            confidence=estimate.confidence,
            breakdown=estimate.breakdown,
        )

    def _should_cache(self, estimate: TokenEstimate, config: Any) -> bool:
        """Decide whether to use caching based on model capabilities.

        Uses explicit caching threshold when available, otherwise falls back
        to implicit threshold. Finally, falls back to 4096 if model is unknown.
        """
        model = str(config.model or "gemini-2.0-flash")
        capabilities = get_model_capabilities(model)
        threshold = 4096
        if capabilities and capabilities.caching:
            if capabilities.caching.explicit_minimum_tokens:
                threshold = int(capabilities.caching.explicit_minimum_tokens)
            elif capabilities.caching.implicit_minimum_tokens:
                threshold = int(capabilities.caching.implicit_minimum_tokens)
        return estimate.max_tokens >= threshold

    def _generate_cache_name(
        self, command: ResolvedCommand, system_instruction: str | None = None
    ) -> str:
        """Generate a short, user-facing cache name from stable fields.

        This is a deterministic summary intended for registry hints and
        human-facing debugging (short SHA). For provider registry lookups,
        use the full `_deterministic_cache_key` below.

        Includes model, prompts, system instruction, and normalized source
        metadata; avoids non-deterministic values such as function object
        addresses in `content_loader`.
        """
        initial = command.initial
        config = initial.config
        model = str(config.model or "gemini-2.0-flash")
        prompts = list(initial.prompts)
        sources = [
            {
                "source_type": s.source_type,
                "identifier": str(s.identifier),
                "mime_type": s.mime_type,
                "size_bytes": s.size_bytes,
            }
            for s in command.resolved_sources
        ]
        payload = {
            "model": model,
            "prompts": prompts,
            "system": system_instruction,
            "sources": sources,
        }
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        content_hash = hashlib.sha256(data).hexdigest()[:12]
        return f"cache_{content_hash}"

    def _deterministic_cache_key(
        self,
        command: ResolvedCommand,
        joined_prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        """Deterministic, full-length key for provider cache registry mapping.

        Captures a stable signature of model, prompt, system, and sources.
        Pure and independent of any provider SDK/runtime. Unlike
        `_generate_cache_name`, this uses the full hash digest to avoid
        collisions in programmatic lookups.
        """
        initial = command.initial
        config = initial.config
        model = str(config.model or "gemini-2.0-flash")
        prompts = [joined_prompt]
        sources = [
            {
                "source_type": s.source_type,
                "identifier": str(s.identifier),
                "mime_type": s.mime_type,
                "size_bytes": s.size_bytes,
            }
            for s in command.resolved_sources
        ]
        payload = {
            "model": model,
            "prompts": prompts,
            "system": system_instruction,
            "sources": sources,
        }
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(data).hexdigest()
