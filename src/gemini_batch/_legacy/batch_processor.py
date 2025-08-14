"""Batch processor for multimodal content analysis"""  # noqa: D415

from contextlib import contextmanager
import logging
from pathlib import Path
import time
from typing import Any, Unpack

from .analysis.schema_analyzer import SchemaAnalyzer
from .config import ClientProtocol, GeminiConfig, get_config
from .efficiency import track_efficiency
from .exceptions import BatchProcessingError
from .gemini_client import GeminiClient
from .prompts import BatchPromptBuilder, StructuredPromptBuilder
from .response import ResponseProcessor
from .response.result_builder import ResultBuilder
from .response.types import ProcessingMetrics, ProcessingOptions
from .telemetry import TelemetryContext, TelemetryContextProtocol

log = logging.getLogger(__name__)


class BatchProcessor:
    """A simple batch processor for processing questions against content."""

    def __init__(
        self,
        _client: ClientProtocol | None = None,
        telemetry_context: TelemetryContextProtocol | None = None,
        **config: Unpack[GeminiConfig],
    ):
        """Creates a processor using ambient configuration.

        Examples:
            processor = BatchProcessor()  # Zero config (uses env vars)
            processor = BatchProcessor(model="gemini-2.5-pro") # Override a setting
            processor = BatchProcessor(_client=my_mock_client) # Advanced: custom client
        """
        if _client and config:
            raise ValueError(
                "Cannot specify both `_client` and config keyword arguments.",
            )

        self.tele = telemetry_context or TelemetryContext()

        if _client:
            self.client = _client
        else:
            # Create a client, passing the telemetry context and any config
            # overrides. This ensures the entire chain shares the same context.
            self.client = GeminiClient(telemetry_context=self.tele, **config)

        self.response_processor = ResponseProcessor()
        self.result_builder = ResultBuilder(self._calculate_efficiency)
        self.schema_analyzer = SchemaAnalyzer()

    @contextmanager
    def _metrics_tracker(self, call_count: int = 1):
        """Context manager for tracking processing metrics"""  # noqa: D415
        metrics = ProcessingMetrics(calls=call_count)
        start_time = time.time()
        try:
            yield metrics
        finally:
            metrics.time = time.time() - start_time

    def _extract_and_track_response(self, response, question_count, metrics, config):
        """Extract answers from response and track usage metrics"""  # noqa: D415
        extraction_result = self.response_processor.extract_answers_from_response(
            response,
            question_count,
            config.response_schema,
        )

        usage_metrics = extraction_result.usage
        metrics.prompt_tokens += usage_metrics.get("prompt_tokens", 0)
        metrics.output_tokens += usage_metrics.get("output_tokens", 0)
        metrics.cached_tokens += usage_metrics.get("cached_tokens", 0)

        return extraction_result

    def process_questions(
        self,
        content: str | Path | list[str | Path],
        questions: list[str],
        compare_methods: bool = False,  # noqa: FBT001, FBT002
        response_schema: Any | None = None,
        client: GeminiClient | None = None,
        **kwargs,  # Accept additional parameters for flexibility
    ) -> dict[str, Any]:
        """Process a list of questions against content.

        Args:
            content: Content to process (text, files, URLs, etc.)
            questions: List of questions to ask
            compare_methods: Whether to run individual comparison
            response_schema: Optional schema for structured output
            client: Optional client override
            **kwargs: Additional parameters passed to underlying processors
        """
        log.info("Starting question processing for %d questions.", len(questions))
        if client:
            self.client = client

        config = ProcessingOptions(
            compare_methods=compare_methods,
            response_schema=response_schema,
            return_usage=kwargs.pop("return_usage", False),  # Extract return_usage flag
            options=kwargs,
        )

        return self._process_standard(content, questions, config)

    def process_questions_multi_source(
        self,
        sources: list[str | Path | list[str | Path]],
        questions: list[str],
        response_schema: Any | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Process multiple sources in a single batch for maximum efficiency"""  # noqa: D415
        if not sources:
            raise ValueError("At least one source is required")
        if not questions:
            raise ValueError("Questions are required")

        result = self.process_questions(
            sources,
            questions,
            response_schema=response_schema,
            **kwargs,
        )

        # Enhance with multi-source metadata
        result.update(
            {
                "source_count": len(sources),
                "processing_mode": "multi_source_batch",
            },
        )

        return result

    def _process_standard(
        self,
        content: str | Path | list[str | Path],
        questions: list[str],
        config: ProcessingOptions,
    ) -> dict[str, Any]:
        """Standard processing path with explicit fallback and comparison logic.

        This method has clear, linear control flow:
        1. Try batch processing first
        2. If batch fails, explicitly fall back to individual processing
        3. If comparison requested and batch succeeded, run individual for comparison
        4. Build and return results
        """
        with self.tele(
            "batch.total_processing",
            question_count=len(questions),
            content_type=type(content).__name__,
        ):
            batch_answers = []
            batch_metrics = ProcessingMetrics.empty()
            individual_answers = None
            individual_metrics = ProcessingMetrics.empty()

            # Primary execution path
            try:
                with self.tele("batch.attempt") as ctx:  # noqa: F841
                    log.info("Attempting batch processing.")
                    batch_answers, batch_metrics = self._process_batch(
                        content,
                        questions,
                        config,
                    )
                    # Contextual metrics - track token efficiency
                    if batch_metrics.total_tokens > 0:
                        self.tele.metric(
                            "token_efficiency",
                            batch_metrics.total_tokens / 1000,
                        )
                    batch_succeeded = True

            except BatchProcessingError as e:
                with self.tele(
                    "batch.individual_fallback", error_type=type(e).__name__
                ):
                    # Batch processing failed - explicitly fall back to individual processing
                    log.warning(
                        "Batch processing failed, falling back to individual calls. Reason: %s",
                        e,
                    )
                    individual_answers, individual_metrics = self._process_individual(
                        content,
                        questions,
                        config,
                    )

                    # Use individual results as the primary result
                    batch_answers = individual_answers
                    batch_metrics = individual_metrics

                    # Clear individual results since they're now the primary result
                    individual_answers = None
                    individual_metrics = ProcessingMetrics.empty()
                    batch_succeeded = False

            # Comparison run (if requested and batch succeeded)
            if config.compare_methods and batch_succeeded:
                with self.tele("batch.comparison_run"):
                    log.debug(
                        "Comparison mode enabled, running individual processing for metrics.",
                    )
                    # Only run comparison if batch processing actually succeeded
                    individual_answers, individual_metrics = self._process_individual(
                        content,
                        questions,
                        config,
                    )

            # Result building
            with self.tele("batch.result_building"):
                return self.result_builder.build_standard_result(
                    questions,
                    batch_answers,  # The definitive answers (batch or fallback)
                    batch_metrics,  # The metrics for the definitive answers
                    individual_metrics,  # Populated ONLY if comparison was run
                    individual_answers,  # Populated ONLY if comparison was run
                    config,
                )

    def _process_batch(
        self,
        content: str | Path | list[str | Path],
        questions: list[str],
        config: ProcessingOptions,
    ) -> tuple[list[str], ProcessingMetrics]:
        """Process all questions in a single batch call."""
        with (
            self.tele(
                "batch.processing",
                method="batch",
                schema_enabled=config.response_schema is not None,
            ),
            self._metrics_tracker(call_count=1) as metrics,
        ):
            # 1. Select prompt builder and analyze schema if needed
            if config.response_schema:
                with self.tele("batch.schema_analysis"):
                    self.schema_analyzer.analyze(config.response_schema)
                    prompt_builder = StructuredPromptBuilder(config.response_schema)
                    log.debug("Using StructuredPromptBuilder for response schema.")
            else:
                prompt_builder = BatchPromptBuilder()
                log.debug("Using BatchPromptBuilder.")

            # 2. Create the batch prompt
            with self.tele("batch.prompt_creation"):
                batch_prompt = prompt_builder.create_prompt(questions)

            try:
                # 3. Make the API call - use generate_content since we have a single combined prompt
                with self.tele("batch.api_call") as api_ctx:  # noqa: F841
                    # Always request usage internally, but filter user's return_usage to avoid conflicts
                    api_options = {
                        k: v for k, v in config.options.items() if k != "return_usage"
                    }
                    response = self.client.generate_content(
                        content=content,
                        prompt=batch_prompt,  # Single combined prompt, not multiple questions
                        return_usage=True,  # Need full response object for ResponseProcessor
                        response_schema=config.response_schema,
                        **api_options,
                    )
                    # Track API call success
                    self.tele.metric("api_calls_successful", 1)

                # 4. Process the successful response
                with self.tele("batch.response_processing"):
                    extraction_result = self._extract_and_track_response(
                        response,
                        len(questions),
                        metrics,
                        config,
                    )

                if extraction_result.structured_quality:
                    metrics.structured_output = extraction_result.structured_quality

                # Ensure we return List[str] for batch processing
                answers = (
                    extraction_result.answers
                    if extraction_result.is_batch_result
                    else [extraction_result.answers]
                )

                # Track successful batch processing
                self.tele.metric("batch_success", 1)
                return answers, metrics

            except Exception as e:
                # Track failed API calls
                self.tele.metric("batch_failures", 1)
                # Any failure in batch processing should be re-raised as BatchProcessingError
                # This makes the failure explicit and allows the caller to handle it appropriately
                raise BatchProcessingError(f"Batch API call failed: {e!s}") from e

    def _process_individual(
        self,
        content: str | Path | list[str | Path],
        questions: list[str],
        config: ProcessingOptions,
    ) -> tuple[list[str], ProcessingMetrics]:
        """Process questions individually.

        This method is used either as a fallback when batch processing fails,
        or for comparison when compare_methods=True.
        """
        with (
            self.tele(
                "individual.processing",
                method="individual",
                question_count=len(questions),
            ),
            self._metrics_tracker(call_count=len(questions)) as metrics,
        ):
            answers = []

            for i, question in enumerate(questions):
                with self.tele(f"individual.question_{i + 1}"):
                    try:
                        # Always request usage internally, but filter user's return_usage to avoid conflicts
                        api_options = {
                            k: v
                            for k, v in config.options.items()
                            if k != "return_usage"
                        }
                        response = self.client.generate_content(
                            content=content,
                            prompt=question,
                            return_usage=True,  # Need full response object for ResponseProcessor
                            response_schema=config.response_schema,
                            **api_options,
                        )

                        extraction_result = self._extract_and_track_response(
                            response,
                            1,
                            metrics,
                            config,
                        )

                        # For individual processing, we always get a single answer (str)
                        answer = (
                            extraction_result.answers
                            if not extraction_result.is_batch_result
                            else extraction_result.answers[0]
                        )
                        answers.append(answer)

                        # Track successful individual calls
                        self.tele.metric("individual_success", 1)

                    except Exception as e:
                        # For individual processing, we can continue with other questions
                        # even if one fails
                        answers.append(f"Error: {e!s}")
                        self.tele.metric("individual_failures", 1)

            return answers, metrics

    def _calculate_efficiency(
        self,
        individual_metrics: ProcessingMetrics,
        batch_metrics: ProcessingMetrics,
    ) -> dict[str, Any]:
        """Calculate efficiency metrics between individual and batch processing"""  # noqa: D415
        return track_efficiency(
            individual_calls=individual_metrics.calls,
            batch_calls=batch_metrics.calls,
            individual_prompt_tokens=individual_metrics.prompt_tokens,
            individual_output_tokens=individual_metrics.output_tokens,
            individual_cached_tokens=individual_metrics.cached_tokens,
            batch_prompt_tokens=batch_metrics.prompt_tokens,
            batch_output_tokens=batch_metrics.output_tokens,
            batch_cached_tokens=batch_metrics.cached_tokens,
            individual_time=individual_metrics.time,
            batch_time=batch_metrics.time,
            include_cache_metrics=True,
        )

    # Cache management methods
    def get_cache_efficiency_summary(self) -> dict[str, Any] | None:
        """Get cache efficiency summary from the underlying client"""  # noqa: D415
        if hasattr(self.client, "get_cache_metrics"):
            return self.client.get_cache_metrics()
        return None

    def cleanup_caches(self) -> int:
        """Clean up expired caches"""  # noqa: D415
        if hasattr(self.client, "cleanup_expired_caches"):
            return self.client.cleanup_expired_caches()
        return 0

    def get_config_summary(self) -> dict[str, Any]:
        """Get configuration summary for debugging"""  # noqa: D415
        summary = {
            "config": get_config(),
            "client_model": getattr(self.client.config_manager, "model", "unknown"),
            "client_caching": getattr(self.client, "config", {}).get(
                "enable_caching",
                False,
            ),
        }

        # Add client-specific information if available
        if hasattr(self.client, "get_config_summary"):
            summary["client_details"] = self.client.get_config_summary()

        return summary
