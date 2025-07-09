"""
Batch processor for multimodal content analysis
"""

from contextlib import contextmanager
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .analysis.schema_analyzer import SchemaAnalyzer
from .config import ConfigManager
from .efficiency import track_efficiency
from .exceptions import BatchProcessingError
from .gemini_client import GeminiClient
from .prompts import BatchPromptBuilder, StructuredPromptBuilder
from .response import ResponseProcessor
from .response.result_builder import ResultBuilder
from .response.types import ProcessingMetrics, ProcessingOptions

log = logging.getLogger(__name__)


class BatchProcessor:
    """Process multiple questions efficiently using batch operations"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ConfigManager] = None,
        client: Optional[Any] = None,
    ):
        self.config = config or ConfigManager()
        self.client = (
            client
            if client
            else GeminiClient.from_env(api_key=api_key or self.config.api_key)
        )
        self.response_processor = ResponseProcessor()
        self.result_builder = ResultBuilder(self._calculate_efficiency)
        self.schema_analyzer = SchemaAnalyzer()

    @contextmanager
    def _metrics_tracker(self, call_count: int = 1):
        """Context manager for tracking processing metrics"""
        metrics = ProcessingMetrics(calls=call_count)
        start_time = time.time()
        try:
            yield metrics
        finally:
            metrics.time = time.time() - start_time

    def _extract_and_track_response(self, response, question_count, metrics, config):
        """Extract answers from response and track usage metrics"""
        extraction_result = self.response_processor.extract_answers_from_response(
            response, question_count, config.response_schema
        )

        usage_metrics = extraction_result.usage
        metrics.prompt_tokens += usage_metrics.get("prompt_tokens", 0)
        metrics.output_tokens += usage_metrics.get("output_tokens", 0)
        metrics.cached_tokens += usage_metrics.get("cached_tokens", 0)

        return extraction_result

    def process_questions(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        questions: List[str],
        compare_methods: bool = False,
        response_schema: Optional[Any] = None,
        client: Optional[GeminiClient] = None,
    ) -> Dict[str, Any]:
        """
        Process a list of questions against a body of text content.
        This method orchestrates the entire process, from content processing
        to response extraction and result building.
        """
        log.info("Starting question processing for %d questions.", len(questions))
        if client:
            self.client = client

        config = ProcessingOptions(
            compare_methods=compare_methods,
            response_schema=response_schema,
        )

        return self._process_standard(content, questions, config)

    def process_questions_multi_source(
        self,
        sources: List[Union[str, Path, List[Union[str, Path]]]],
        questions: List[str],
        response_schema: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process multiple sources in a single batch for maximum efficiency"""
        if not sources:
            raise ValueError("At least one source is required")
        if not questions:
            raise ValueError("Questions are required")

        result = self.process_questions(
            sources,  # Combined content handled by unified client
            questions,
            response_schema=response_schema,
            **kwargs,
        )

        # Enhance with multi-source metadata
        result.update(
            {
                "source_count": len(sources),
                "processing_mode": "multi_source_batch",
            }
        )

        return result

    def _process_standard(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        questions: List[str],
        config: ProcessingOptions,
    ) -> Dict[str, Any]:
        """
        Standard processing path with explicit fallback and comparison logic.

        This method has clear, linear control flow:
        1. Try batch processing first
        2. If batch fails, explicitly fall back to individual processing
        3. If comparison requested and batch succeeded, run individual for comparison
        4. Build and return results
        """
        batch_answers = []
        batch_metrics = ProcessingMetrics.empty()
        individual_answers = None
        individual_metrics = ProcessingMetrics.empty()

        # --- Primary Execution Path ---
        try:
            # Attempt batch processing first - this either succeeds or raises BatchProcessingError
            log.info("Attempting batch processing.")
            batch_answers, batch_metrics = self._process_batch(
                content, questions, config
            )
            batch_succeeded = True

        except BatchProcessingError as e:
            # Batch processing failed - explicitly fall back to individual processing
            log.warning(
                "Batch processing failed, falling back to individual calls. Reason: %s",
                e,
            )
            individual_answers, individual_metrics = self._process_individual(
                content, questions, config
            )

            # Use individual results as the primary result
            batch_answers = individual_answers
            batch_metrics = individual_metrics

            # Clear individual results since they're now the primary result
            individual_answers = None
            individual_metrics = ProcessingMetrics.empty()
            batch_succeeded = False

        # --- Comparison Run (if requested and batch succeeded) ---
        if config.compare_methods and batch_succeeded:
            log.debug(
                "Comparison mode enabled, running individual processing for metrics."
            )
            # Only run comparison if batch processing actually succeeded
            individual_answers, individual_metrics = self._process_individual(
                content, questions, config
            )

        # --- Result Building ---
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
        content: Union[str, Path, List[Union[str, Path]]],
        questions: List[str],
        config: ProcessingOptions,
    ) -> Tuple[List[str], ProcessingMetrics]:
        """Process all questions in a single batch call."""
        with self._metrics_tracker(call_count=1) as metrics:
            # 1. Select prompt builder and analyze schema if needed
            if config.response_schema:
                self.schema_analyzer.analyze(config.response_schema)
                prompt_builder = StructuredPromptBuilder(config.response_schema)
                log.debug("Using StructuredPromptBuilder for response schema.")
            else:
                prompt_builder = BatchPromptBuilder()
                log.debug("Using BatchPromptBuilder.")

            # 2. Create the batch prompt
            batch_prompt = prompt_builder.create_prompt(questions)

            try:
                # 3. Make the API call - this is the critical operation that can fail
                response = self.client.generate_batch(
                    content=content,
                    questions=[
                        batch_prompt
                    ],  # The builder combines questions into one prompt
                    return_usage=False,  # Let ResponseProcessor handle usage extraction
                    response_schema=config.response_schema,
                    **config.options,
                )

                # 4. Process the successful response
                extraction_result = self._extract_and_track_response(
                    response, len(questions), metrics, config
                )

                if extraction_result.structured_quality:
                    metrics.structured_output = extraction_result.structured_quality

                # Ensure we return List[str] for batch processing
                answers = (
                    extraction_result.answers
                    if extraction_result.is_batch_result
                    else [extraction_result.answers]
                )

                return answers, metrics

            except Exception as e:
                # Any failure in batch processing should be re-raised as BatchProcessingError
                # This makes the failure explicit and allows the caller to handle it appropriately
                raise BatchProcessingError(f"Batch API call failed: {str(e)}") from e

    def _process_individual(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        questions: List[str],
        config: ProcessingOptions,
    ) -> Tuple[List[str], ProcessingMetrics]:
        """
        Process questions individually.

        This method is used either as a fallback when batch processing fails,
        or for comparison when compare_methods=True.
        """
        with self._metrics_tracker(call_count=len(questions)) as metrics:
            answers = []

            for question in questions:
                try:
                    response = self.client.generate_content(
                        content=content,
                        prompt=question,
                        return_usage=False,  # Let ResponseProcessor handle usage extraction
                        response_schema=config.response_schema,
                        **config.options,
                    )

                    extraction_result = self._extract_and_track_response(
                        response, 1, metrics, config
                    )

                    # For individual processing, we always get a single answer (str)
                    answer = (
                        extraction_result.answers
                        if not extraction_result.is_batch_result
                        else extraction_result.answers[0]
                    )
                    answers.append(answer)

                except Exception as e:
                    # For individual processing, we can continue with other questions
                    # even if one fails
                    answers.append(f"Error: {str(e)}")

            return answers, metrics

    def _calculate_efficiency(
        self, individual_metrics: ProcessingMetrics, batch_metrics: ProcessingMetrics
    ) -> Dict[str, Any]:
        """Calculate efficiency metrics between individual and batch processing"""
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
    def get_cache_efficiency_summary(self) -> Optional[Dict[str, Any]]:
        """Get cache efficiency summary from the underlying client"""
        if hasattr(self.client, "get_cache_metrics"):
            return self.client.get_cache_metrics()
        return None

    def cleanup_caches(self) -> int:
        """Clean up expired caches"""
        if hasattr(self.client, "cleanup_expired_caches"):
            return self.client.cleanup_expired_caches()
        return 0
