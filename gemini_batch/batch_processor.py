"""
Batch processor for text content analysis
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .analysis.schema_analyzer import SchemaAnalyzer
from .efficiency import track_efficiency
from .gemini_client import GeminiClient
from .prompts import BatchPromptBuilder, StructuredPromptBuilder
from .response import ResponseProcessor


@dataclass
class ProcessingMetrics:
    """Container for processing metrics"""

    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    time: float = 0.0
    structured_output: Optional[Dict[str, Any]] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def effective_tokens(self) -> int:
        """Effective tokens for cost calculation (excluding cached portion)"""
        return (self.prompt_tokens - self.cached_tokens) + self.output_tokens

    @property
    def cache_hit_ratio(self) -> float:
        """Ratio of cached to total prompt tokens"""
        return self.cached_tokens / max(self.prompt_tokens, 1)

    @classmethod
    def empty(cls) -> "ProcessingMetrics":
        """Create empty metrics for comparison baseline"""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility"""
        result = {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "tokens": self.total_tokens,
            "effective_tokens": self.effective_tokens,
            "cache_hit_ratio": self.cache_hit_ratio,
            "time": self.time,
        }
        if self.structured_output:
            result["structured_output"] = self.structured_output
        return result


@dataclass
class ProcessingOptions:
    """Container for processing configuration"""

    compare_methods: bool = False
    return_usage: bool = False
    response_schema: Optional[Any] = None
    options: Dict[str, Any] = field(default_factory=dict)


class ResultBuilder:
    """Unified result building for different processing modes"""

    def __init__(self, efficiency_calculator):
        self.efficiency_calculator = efficiency_calculator

    def build_standard_result(
        self,
        questions: List[str],
        batch_answers: List[str],
        batch_metrics: ProcessingMetrics,
        individual_metrics: ProcessingMetrics,
        individual_answers: Optional[List[str]],
        config: ProcessingOptions,
    ) -> Dict[str, Any]:
        """Build result for standard processing mode"""
        # Check if batch processing failed (0 calls) and we have individual metrics
        if batch_metrics.calls == 0 and individual_metrics.calls > 0:
            # Batch failed, use individual metrics as primary
            efficiency = self.efficiency_calculator(individual_metrics, batch_metrics)

            result = {
                "question_count": len(questions),
                "answers": batch_answers,  # These are actually individual answers
                "efficiency": efficiency,
                "metrics": {
                    "batch": batch_metrics.to_dict(),
                    "individual": individual_metrics.to_dict(),
                },
            }
        else:
            # Normal batch processing
            efficiency = self.efficiency_calculator(individual_metrics, batch_metrics)

            result = {
                "question_count": len(questions),
                "answers": batch_answers,
                "efficiency": efficiency,
                "metrics": {
                    "batch": batch_metrics.to_dict(),
                    "individual": individual_metrics.to_dict(),
                },
            }

        # Add cache summary if caching was used
        if batch_metrics.cached_tokens > 0 or individual_metrics.cached_tokens > 0:
            result["cache_summary"] = {
                "cache_enabled": True,
                "batch_cache_hit_ratio": batch_metrics.cache_hit_ratio,
                "individual_cache_hit_ratio": individual_metrics.cache_hit_ratio,
                "tokens_saved": individual_metrics.total_tokens
                - batch_metrics.total_tokens,
                "cache_cost_benefit": efficiency.get("cache_efficiency", {}).get(
                    "cache_improvement_factor", 1.0
                ),
            }
        else:
            result["cache_summary"] = {"cache_enabled": False}

        self._add_optional_data(result, batch_metrics, individual_answers, config)
        return result

    def enhance_response_processor_result(
        self,
        result: Dict[str, Any],
        batch_metrics: ProcessingMetrics,
        individual_metrics: ProcessingMetrics,
        individual_answers: List[str],
    ) -> None:
        """Add efficiency comparison metrics to ResponseProcessor result"""
        efficiency = self.efficiency_calculator(individual_metrics, batch_metrics)

        result.update(
            {
                "efficiency": efficiency,
                "metrics": {
                    "batch": batch_metrics.to_dict(),
                    "individual": individual_metrics.to_dict(),
                },
                "individual_answers": individual_answers,
            }
        )

    def _add_optional_data(
        self,
        result: Dict[str, Any],
        batch_metrics: ProcessingMetrics,
        individual_answers: Optional[List[str]],
        config: ProcessingOptions,
    ) -> None:
        """Add optional data to result dictionary"""
        # Add structured data if available
        if batch_metrics.structured_output and batch_metrics.structured_output.get(
            "structured_data"
        ):
            result["structured_data"] = batch_metrics.structured_output[
                "structured_data"
            ]

        # Add individual answers if comparison was requested
        if config.compare_methods and individual_answers:
            result["individual_answers"] = individual_answers

        # Add usage information if requested
        if config.return_usage:
            result["usage"] = {
                "prompt_tokens": batch_metrics.prompt_tokens,
                "output_tokens": batch_metrics.output_tokens,
                "total_tokens": batch_metrics.total_tokens,
                "cached_tokens": batch_metrics.cached_tokens,
                "effective_tokens": batch_metrics.effective_tokens,
                "cache_hit_ratio": batch_metrics.cache_hit_ratio,
            }


class BatchProcessor:
    """Process multiple questions efficiently using batch operations"""

    def __init__(self, client: Optional[GeminiClient] = None, **client_kwargs):
        """Initialize with unified client"""
        if client is not None:
            self.client = client
        elif client_kwargs:
            # Use from_env for both cases - it handles direct API key parameters
            self.client = GeminiClient.from_env(**client_kwargs)
        else:
            # No arguments provided - use environment
            self.client = GeminiClient.from_env()

        self.response_processor = ResponseProcessor()
        self.result_builder = ResultBuilder(self._calculate_efficiency)
        self.schema_analyzer = SchemaAnalyzer()

        # Initialize metrics tracking for testing
        self.individual_calls = 0

    @contextmanager
    def _metrics_tracker(self, call_count: int = 1):
        """Context manager for tracking processing metrics"""
        metrics = ProcessingMetrics(calls=call_count)
        start_time = time.time()
        try:
            yield metrics
        finally:
            metrics.time = time.time() - start_time

    def process_questions(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        questions: List[str],
        compare_methods: bool = False,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Dict[str, Any]:
        """Unified method to process questions about any content type"""
        if not questions:
            raise ValueError("Questions are required")

        config = ProcessingOptions(
            compare_methods=compare_methods,
            return_usage=return_usage,
            response_schema=response_schema,
            options=options,
        )

        # Always use standard processing with integrated ResponseProcessor
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
        """Standard processing path using integrated ResponseProcessor"""
        # Process batch first
        batch_answers, batch_metrics = self._process_batch(content, questions, config)

        # Optionally process individually for comparison
        individual_metrics = ProcessingMetrics.empty()
        individual_answers = None
        batch_failed = False

        # If batch_metrics is actually from individual fallback, assign to individual
        if batch_metrics.calls == len(questions) and batch_metrics.prompt_tokens > 0:
            # This is a fallback case - individual processing was used
            individual_metrics = batch_metrics
            batch_metrics = ProcessingMetrics.empty()
            individual_answers = batch_answers
            # Do not replace batch_answers with None; keep the actual answers for the result
            batch_failed = True

        if config.compare_methods and not batch_failed:
            individual_answers, individual_metrics = self._process_individual(
                content, questions, config
            )

        # Build and return result
        return self.result_builder.build_standard_result(
            questions,
            batch_answers,
            batch_metrics,
            individual_metrics,
            individual_answers,
            config,
        )

    def _process_batch(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        questions: List[str],
        config: ProcessingOptions,
    ) -> Tuple[List[str], ProcessingMetrics]:
        """Process all questions in a single batch call"""
        with self._metrics_tracker(call_count=1) as metrics:
            # 1. Select prompt builder and (optionally) analyze schema
            if config.response_schema:
                self.schema_analyzer.analyze(config.response_schema)
                prompt_builder = StructuredPromptBuilder(config.response_schema)
            else:
                prompt_builder = BatchPromptBuilder()

            # 2. Create the prompt text
            batch_prompt = prompt_builder.create_prompt(questions)

            try:
                # 3. Call client, passing the schema
                response = self.client.generate_batch(
                    content=content,
                    questions=[
                        batch_prompt
                    ],  # The builder combines questions into one prompt
                    return_usage=False,  # Let ResponseProcessor handle usage extraction
                    response_schema=config.response_schema,  # Pass schema down
                    **config.options,
                )

                # 4. Process the response (which is now more reliable)
                extraction_result = (
                    self.response_processor.extract_answers_from_response(
                        response, len(questions), config.response_schema
                    )
                )

                usage_metrics = extraction_result.usage
                metrics.prompt_tokens = usage_metrics.get("prompt_tokens", 0)
                metrics.output_tokens = usage_metrics.get("output_tokens", 0)
                metrics.cached_tokens = usage_metrics.get("cached_tokens", 0)

                if extraction_result.structured_quality:
                    metrics.structured_output = extraction_result.structured_quality

                # Ensure we return List[str] for batch processing
                answers = (
                    extraction_result.answers
                    if extraction_result.is_batch_result
                    else [extraction_result.answers]
                )
                return answers, metrics

            except Exception:
                # Fallback to individual processing
                individual_answers, individual_metrics = self._process_individual(
                    content, questions, config
                )

                # Return individual answers but with individual metrics (not batch metrics)
                return individual_answers, individual_metrics

    def _process_individual(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        questions: List[str],
        config: ProcessingOptions,
    ) -> Tuple[List[str], ProcessingMetrics]:
        """Process questions individually for comparison or fallback"""
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

                    extraction_result = (
                        self.response_processor.extract_answers_from_response(
                            response,
                            1,
                            config.response_schema,  # Single question
                        )
                    )

                    # For individual processing, we always get a single answer (str)
                    answer = (
                        extraction_result.answers
                        if not extraction_result.is_batch_result
                        else extraction_result.answers[0]
                    )
                    answers.append(answer)

                    usage_metrics = extraction_result.usage
                    metrics.prompt_tokens += usage_metrics.get("prompt_tokens", 0)
                    metrics.output_tokens += usage_metrics.get("output_tokens", 0)
                    metrics.cached_tokens += usage_metrics.get("cached_tokens", 0)

                except Exception as e:
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
