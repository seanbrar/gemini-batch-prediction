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
    RateConstraint,
    ResolvedCommand,
    Result,
    Success,
    TextPart,
    TokenEstimate,
    UploadTask,
)
from gemini_batch.pipeline.base import BaseAsyncHandler
from gemini_batch.pipeline.tokens.adapters.gemini import (
    GeminiEstimationAdapter,
)
from gemini_batch.telemetry import TelemetryContext, TelemetryContextProtocol

if TYPE_CHECKING:
    from gemini_batch.config import GeminiConfig

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
        if estimation_adapter is not None:
            self._adapter: Any = estimation_adapter
        else:
            # Instantiate the default Gemini estimation adapter eagerly
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
            model_name: str = str(initial.config.get("model") or "gemini-2.0-flash")

            # Validate the prompts exist to avoid invalid states.
            if not initial.prompts:
                return Failure(ConfigurationError("At least one prompt is required."))

            # Assemble a minimal text payload from prompts. The API layer will
            # translate these neutral parts to provider-specific objects.
            joined_prompt = "\n\n".join(initial.prompts)

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
            if self._should_cache(total_estimate, initial.config):
                key = self._deterministic_cache_key(command, joined_prompt)
                # In this minimal slice, cache the prompt (part index 0) and include system_instruction when present later
                explicit_cache = ExplicitCachePlan(
                    create=True,
                    cache_name=None,
                    contents_part_indexes=(0,),
                    include_system_instruction=True,
                    ttl_seconds=(
                        int(cast("int | str", initial.config.get("ttl_seconds")))
                        if initial.config.get("ttl_seconds") is not None
                        else None
                    ),
                    deterministic_key=key,
                )
                # For backward compatibility with tests expecting a cache hint
                # provide a deterministic cache_name_to_use hint.
                cache_hint = self._generate_cache_name(command)
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

            api_call = APICall(
                model_name=model_name,
                api_parts=tuple(parts),
                api_config={},
                cache_name_to_use=cache_hint,
            )

            # Resolve rate limits (vendor-neutral via core.models).
            # Planner degrades gracefully when tier/model is unknown/unavailable.
            rate_constraint: RateConstraint | None = None
            try:
                from gemini_batch.core.models import APITier, get_rate_limits

                tier_value = str(initial.config.get("tier", "free")).upper()
                tier = (
                    APITier[tier_value]
                    if tier_value in APITier.__members__
                    else APITier.FREE
                )
                limits = get_rate_limits(tier, model_name)
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
    def _should_cache(self, estimate: TokenEstimate, config: GeminiConfig) -> bool:
        """Decide whether to use caching based on model capabilities.

        Uses explicit caching threshold when available, otherwise falls back
        to implicit threshold. Finally, falls back to 4096 if model is unknown.
        """
        model = str(config.get("model", "gemini-2.0-flash"))
        capabilities = get_model_capabilities(model)
        threshold = 4096
        if capabilities and capabilities.caching:
            if capabilities.caching.explicit_minimum_tokens:
                threshold = int(capabilities.caching.explicit_minimum_tokens)
            elif capabilities.caching.implicit_minimum_tokens:
                threshold = int(capabilities.caching.implicit_minimum_tokens)
        return estimate.max_tokens >= threshold

    def _generate_cache_name(self, command: ResolvedCommand) -> str:
        """Generate a deterministic cache key based on stable fields.

        Includes model, prompts, and normalized source metadata, avoiding
        non-deterministic function object addresses in `content_loader`.
        """
        initial = command.initial
        model = str(initial.config.get("model", "gemini-2.0-flash"))
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
        payload = {"model": model, "prompts": prompts, "sources": sources}
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        content_hash = hashlib.sha256(data).hexdigest()[:12]
        return f"cache_{content_hash}"

    def _deterministic_cache_key(
        self, command: ResolvedCommand, joined_prompt: str
    ) -> str:
        """Deterministic key capturing model + stable source + prompt signature.

        This is used by registries to map to provider cache names. It is pure and
        independent of any provider SDK or runtime state.
        """
        initial = command.initial
        model = str(initial.config.get("model", "gemini-2.0-flash"))
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
        payload = {"model": model, "prompts": prompts, "sources": sources}
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(data).hexdigest()
