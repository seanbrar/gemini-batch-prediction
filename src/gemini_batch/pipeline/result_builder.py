"""Result builder utilities for converting raw API responses to results.

Provides the Two-Tier Transform Chain used to produce a stable
`ResultEnvelope` from a `FinalizedCommand`: a prioritized set of
transforms (Tier 1) with a `MinimalProjection` fallback (Tier 2).

Focus: how to configure and call `ResultBuilder`,what it returns,
and when diagnostics are produced.
"""

from dataclasses import asdict
import time
from typing import Any, Never

from gemini_batch.core.types import (
    FinalizedCommand,
    Result,
    Success,
)
from gemini_batch.pipeline.base import BaseAsyncHandler
from gemini_batch.pipeline.results.extraction import (
    ExtractionContext,
    ExtractionContract,
    ExtractionDiagnostics,
    ExtractionResult,
    TransformSpec,
    Violation,
)
from gemini_batch.pipeline.results.minimal_projection import MinimalProjection
from gemini_batch.pipeline.results.transforms import default_transforms


class ResultBuilder(BaseAsyncHandler[FinalizedCommand, dict[str, Any], Never]):
    """Build `ResultEnvelope` objects from finalized commands.

    The `ResultBuilder` applies configured `TransformSpec`s in priority order
    and falls back to `MinimalProjection` to guarantee a deterministic,
    non-failing result. It can optionally collect extraction diagnostics.

    Attributes:
        transforms: Tuple of `TransformSpec` used by Tier 1 extraction.
        enable_diagnostics: Whether to collect `ExtractionDiagnostics`.
        max_text_size: Maximum response text size processed.
    """

    def __init__(
        self,
        transforms: tuple[TransformSpec, ...] | None = None,
        *,
        enable_diagnostics: bool = False,
        max_text_size: int = 1_000_000,  # 1MB limit
    ) -> None:
        """Initialize the ResultBuilder.

        Args:
            transforms: Optional sequence of `TransformSpec`. Defaults to built-ins.
            enable_diagnostics: If True, attach `ExtractionDiagnostics` to results.
            max_text_size: Max text length to process; oversized inputs are truncated.
        """
        self.transforms = (
            transforms if transforms is not None else tuple(default_transforms())
        )
        self.enable_diagnostics = enable_diagnostics
        self.max_text_size = max_text_size
        self._minimal_projection = MinimalProjection()
        # Pre-compute deterministic transform order (higher priority first, name tiebreaker)
        self._sorted_transforms: tuple[TransformSpec, ...] = tuple(
            sorted(self.transforms, key=lambda t: (-t.priority, t.name))
        )

    async def handle(self, command: FinalizedCommand) -> Result[dict[str, Any], Never]:
        """Extract a `ResultEnvelope` from a `FinalizedCommand`.

        The method applies Tier 1 transforms (priority order) and a Tier 2
        fallback (`MinimalProjection`) if none match. It performs record-only
        schema and contract validation and attaches diagnostics when enabled.

        Args:
            command: `FinalizedCommand` containing the raw API response.

        Returns:
            `Success` carrying a `ResultEnvelope` (dict). This method does not
            raise for extraction failures; validation issues are recorded.
        """
        start_time = time.perf_counter()

        raw = command.raw_api_response
        ctx = self._build_extraction_context(command)
        diagnostics = ExtractionDiagnostics() if self.enable_diagnostics else None

        # Truncate oversized responses
        if self._is_oversized(raw):
            raw = self._truncate_response(raw)
            if diagnostics:
                diagnostics.flags.add("truncated_input")

        # Tier 1: Try transforms in priority order (with stable name sort for ties)
        extraction_result = None
        for transform in self._sorted_transforms:
            if diagnostics:
                diagnostics.attempted_transforms.append(transform.name)

            if transform.matcher(raw):
                try:
                    extracted_data = transform.extractor(raw, ctx.config)
                    extraction_result = self._create_extraction_result(
                        extracted_data, transform.name
                    )
                    if diagnostics:
                        diagnostics.successful_transform = transform.name
                    break
                except Exception as e:
                    if diagnostics:
                        diagnostics.transform_errors[transform.name] = str(e)
                    continue  # Try next transform

        # Tier 2: Minimal Projection fallback (always succeeds)
        if extraction_result is None:
            fallback_result = self._minimal_projection.extract(raw, ctx)
            extraction_result = fallback_result
            if diagnostics:
                diagnostics.successful_transform = fallback_result.method

        # Record pre-normalization answer count for validation messages
        original_answer_count = len(extraction_result.answers)

        # Build result envelope
        result_envelope = self._build_result_envelope(extraction_result, command, ctx)

        # Schema validation (record-only)
        violations = self._validate_schema(result_envelope, ctx)

        # Contract validation (record-only)
        contract = ExtractionContract()
        violations.extend(contract.validate(result_envelope))

        # Add mismatch warning based on original, pre-padding/truncation count
        if original_answer_count != ctx.expected_count:
            violations.insert(
                0,
                Violation(
                    f"Expected {ctx.expected_count} answers, got {original_answer_count}",
                    "warning",
                ),
            )

        # Finalize diagnostics
        if diagnostics:
            end_time = time.perf_counter()
            diagnostics.extraction_duration_ms = (end_time - start_time) * 1000
            diagnostics.contract_violations = violations
            diagnostics.expected_answer_count = ctx.expected_count
            diagnostics.original_answer_count = original_answer_count
            # Convert to plain dict and ensure JSON-serializable fields
            diag_dict = asdict(diagnostics)
            flags = diag_dict.get("flags")
            if isinstance(flags, set):
                diag_dict["flags"] = sorted(flags)
            result_envelope["diagnostics"] = diag_dict
        elif violations:
            # Include warnings even without full diagnostics
            result_envelope["validation_warnings"] = tuple(
                v.message for v in violations
            )

        return Success(result_envelope)

    def _build_extraction_context(self, command: FinalizedCommand) -> ExtractionContext:
        """Create an `ExtractionContext` from a `FinalizedCommand`.

        Args:
            command: Planned command used to derive expected answer count
                and prompts.

        Returns:
            `ExtractionContext` with `expected_count`, `prompts`, and config.
        """
        # Get expected count from prompts
        prompts = command.planned.resolved.initial.prompts
        expected_count = len(prompts) if prompts else 1

        return ExtractionContext(
            expected_count=expected_count,
            prompts=prompts,
            config={},  # TODO: Consider surfacing minimal, explicit knobs here
            # (e.g., answer cleaning rules) without leaking provider
            # concerns or introducing implicit behavior.
        )

    def _create_extraction_result(
        self, extracted_data: dict[str, Any], method: str
    ) -> ExtractionResult:
        """Normalize transform output into an `ExtractionResult`.

        Args:
            extracted_data: Raw dict returned by a transform's extractor.
            method: Name of the transform that produced the data.

        Returns:
            `ExtractionResult` with normalized `answers`, `confidence`, and
            optional `structured_data`.
        """
        answers = extracted_data.get("answers", [])
        confidence = extracted_data.get("confidence", 0.5)
        structured_data = extracted_data.get("structured_data")

        # Ensure answers are strings
        string_answers = [str(answer) for answer in answers]

        return ExtractionResult(
            answers=string_answers,
            method=method,
            confidence=confidence,
            structured_data=structured_data,
        )

    def _build_result_envelope(
        self,
        extraction_result: ExtractionResult,
        command: FinalizedCommand,
        ctx: ExtractionContext,
    ) -> dict[str, Any]:
        """Package extraction output into a stable `ResultEnvelope` dict.

        Ensures answer count matches `ctx.expected_count`, inserts
        `structured_data` when available, and safely extracts telemetry
        metrics from `command`.

        Args:
            extraction_result: Normalized extraction result.
            command: Original command (may contain telemetry).
            ctx: Extraction context used for padding/truncation rules.

        Returns:
            `ResultEnvelope` dictionary ready for downstream consumers.
        """
        # Ensure we have the right number of answers
        answers = extraction_result.answers
        if len(answers) < ctx.expected_count:
            # Pad with empty strings
            answers = answers + [""] * (ctx.expected_count - len(answers))
        elif len(answers) > ctx.expected_count:
            # Truncate to expected count
            answers = answers[: ctx.expected_count]

        # Build base envelope
        envelope: dict[str, Any] = {
            "success": True,
            "answers": answers,
            "extraction_method": extraction_result.method,
            "confidence": extraction_result.confidence,
        }

        # Add structured data if available
        if extraction_result.structured_data is not None:
            envelope["structured_data"] = extraction_result.structured_data

        # Integrate telemetry data from command with graceful degradation
        telemetry_data = (
            command.telemetry_data if isinstance(command.telemetry_data, dict) else {}
        )
        durations_obj = telemetry_data.get("durations")
        token_validation_obj = telemetry_data.get("token_validation")
        usage_obj = telemetry_data.get("usage")
        durations: dict[str, Any] = (
            dict(durations_obj) if isinstance(durations_obj, dict) else {}
        )
        token_validation: dict[str, Any] = (
            dict(token_validation_obj) if isinstance(token_validation_obj, dict) else {}
        )
        envelope["metrics"] = {
            "durations": durations,
            "token_validation": token_validation,
        }

        # Include token usage details when provided by telemetry (optional)
        if isinstance(usage_obj, dict) and usage_obj:
            envelope["usage"] = dict(usage_obj)

        return envelope

    def _validate_schema(
        self, result: dict[str, Any], ctx: ExtractionContext
    ) -> list[Violation]:
        """Run record-only schema validation and return violations.

        This records issues for telemetry but does not raise or change the
        extraction outcome.

        Args:
            result: The `ResultEnvelope` to validate.
            ctx: Extraction context that may include a `schema`.

        Returns:
            List of `Violation` objects describing schema issues.
        """
        if ctx.schema is None:
            return []

        violations = []
        try:
            # Attempt Pydantic validation if available
            if hasattr(ctx.schema, "model_validate"):
                # Use structured_data if available, otherwise use answers
                payload = result.get("structured_data") or {
                    "answers": result["answers"]
                }
                ctx.schema.model_validate(payload)
            else:
                violations.append(
                    Violation("Schema is not a Pydantic v2 model", "info")
                )
        except Exception as e:
            violations.append(Violation(f"Schema validation failed: {e}", "warning"))

        return violations

    def _is_oversized(self, raw: Any) -> bool:
        """Return True if `raw` exceeds configured `max_text_size`.

        Args:
            raw: Raw API response to check.
        """
        if isinstance(raw, str):
            return len(raw) > self.max_text_size
        if isinstance(raw, dict):
            # Rough estimate for dict size
            return len(str(raw)) > self.max_text_size
        return False

    def _truncate_response(self, raw: Any) -> Any:
        """Truncate `raw` inputs that exceed `max_text_size`.

        The method preserves structure for dict inputs by truncating long
        string fields; for strings it truncates and appends a marker.
        """
        if isinstance(raw, str):
            if len(raw) > self.max_text_size:
                return raw[: self.max_text_size] + "... [TRUNCATED]"
            return raw
        if isinstance(raw, dict):
            # For dicts, try to truncate text fields
            truncated_dict: dict[str, Any] = {}
            for key, value in raw.items():
                if isinstance(value, str) and len(value) > self.max_text_size:
                    truncated_dict[key] = (
                        value[: self.max_text_size] + "... [TRUNCATED]"
                    )
                else:
                    truncated_dict[key] = value
            return truncated_dict
        return raw
