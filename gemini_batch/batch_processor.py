"""
Batch processor for text content analysis
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .client import GeminiClient
from .efficiency import track_efficiency
from .response import extract_answers


@dataclass
class ProcessingMetrics:
    """Container for processing metrics"""

    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    time: float = 0.0
    structured_output: Optional[Dict[str, Any]] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

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
            "tokens": self.total_tokens,
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
    response_processor: Optional[Any] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result from unified response extraction"""

    answers: Union[
        List[str], str
    ]  # List for batch (multiple questions), str for individual
    usage: Dict[str, int]
    structured_quality: Optional[Dict[str, Any]] = None

    @property
    def is_batch_result(self) -> bool:
        """True if this represents a batch response (multiple answers)"""
        return isinstance(self.answers, list)


class ResponseExtractor:
    """Unified response extraction and processing"""

    def __init__(self):
        pass

    def extract_from_response(
        self, response: Any, question_count: int, response_schema: Optional[Any] = None
    ) -> ExtractionResult:
        """
        Unified response extraction for both batch and individual responses

        This method automatically detects the response type based on question_count:
        - question_count > 1: Treats as batch response, returns List[str] of answers
        - question_count = 1: Treats as individual response, returns single str answer

        Args:
            response: Raw API response object or dict
            question_count: Number of questions asked (determines batch vs individual handling)
            response_schema: Optional schema for structured response validation

        Returns:
            ExtractionResult containing answers, usage metrics, and optional structured quality data
        """
        usage = {"prompt_tokens": 0, "output_tokens": 0}
        structured_quality = None
        is_batch = question_count > 1

        if isinstance(response, dict):
            usage = response.get("usage", usage)

            if response_schema:
                answers, structured_quality = self._handle_structured_response(
                    response, is_batch, question_count
                )
            else:
                if is_batch:
                    response_text = response.get("text", "")
                    answers = extract_answers(
                        response_text, question_count, is_structured=False
                    )
                else:  # Individual response
                    answers = response.get("text", "")
        else:
            # Simple response handling
            if is_batch:
                answers = extract_answers(response, question_count, is_structured=False)
            else:  # Individual response
                answers = str(response) if response is not None else ""

        return ExtractionResult(
            answers=answers, usage=usage, structured_quality=structured_quality
        )

    def _handle_structured_response(
        self, response: Dict[str, Any], is_batch: bool, question_count: int
    ) -> Tuple[Union[List[str], str], Optional[Dict[str, Any]]]:
        """
        Handle structured response parsing for both batch and individual responses

        Args:
            response: Dict response from API
            is_batch: True for batch responses (multiple questions), False for individual
            question_count: Number of questions asked (needed for text extraction fallback)
        """
        if response.get("structured_success"):
            parsed_data = response.get("parsed")

            if is_batch:
                answers = [str(parsed_data)] if parsed_data else ["No structured data"]
            else:  # Individual response
                answers = (
                    str(parsed_data)
                    if parsed_data is not None
                    else response.get("text", "")
                )

            structured_quality = {
                "confidence": response.get("structured_confidence", 0.0),
                "method": response.get("validation_method", "unknown"),
                "errors": response.get("validation_errors", []),
                "structured_data": parsed_data,
            }
            return answers, structured_quality
        else:
            # Fallback to text extraction
            response_text = response.get("text", "")

            if is_batch:
                # For batch, extract multiple answers from text using the correct question count
                answers = extract_answers(
                    response_text,
                    question_count,  # Use the parameter, not response data
                    is_structured=False,
                )
            else:  # Individual response
                answers = response_text

            structured_quality = {
                "confidence": 0.3,  # Lower confidence for text fallback
                "method": "text_fallback",
                "errors": response.get(
                    "validation_errors", ["Structured parsing failed"]
                ),
            }
            return answers, structured_quality


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
                "cached_tokens": 0,
            }


class BatchProcessor:
    """Process multiple questions efficiently using batch operations"""

    def __init__(self, client: Optional[GeminiClient] = None, **client_kwargs):
        """Initialize with unified client"""
        self.client = client or GeminiClient(**client_kwargs)
        self.response_extractor = ResponseExtractor()
        self.result_builder = ResultBuilder(self._calculate_efficiency)

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
        response_processor=None,
        **options,
    ) -> Dict[str, Any]:
        """Unified method to process questions about any content type"""
        if not questions:
            raise ValueError("Questions are required")

        config = ProcessingOptions(
            compare_methods=compare_methods,
            return_usage=return_usage,
            response_schema=response_schema,
            response_processor=response_processor
            or self._create_response_processor_if_needed(response_schema),
            options=options,
        )

        if config.response_processor:
            return self._process_with_response_processor(content, questions, config)

        return self._process_standard(content, questions, config)

    def process_questions_multi_source(
        self,
        sources: List[Union[str, Path, List[Union[str, Path]]]],
        questions: List[str],
        response_schema: Optional[Any] = None,
        response_processor=None,
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
            response_processor=response_processor,
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

    def _create_response_processor_if_needed(self, response_schema: Optional[Any]):
        """Auto-create ResponseProcessor when response_schema is provided"""
        if response_schema is not None:
            from .response import ResponseProcessor

            return ResponseProcessor()
        return None

    def _process_standard(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        questions: List[str],
        config: ProcessingOptions,
    ) -> Dict[str, Any]:
        """Standard processing path without ResponseProcessor"""
        # Process batch first
        batch_answers, batch_metrics = self._process_batch(content, questions, config)

        # Optionally process individually for comparison
        individual_metrics = ProcessingMetrics.empty()
        individual_answers = None

        if config.compare_methods:
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
            try:
                response = self.client.generate_batch(
                    content=content,
                    questions=questions,
                    return_usage=True,
                    response_schema=config.response_schema,
                    **config.options,
                )

                extraction_result = self.response_extractor.extract_from_response(
                    response, len(questions), config.response_schema
                )

                metrics.prompt_tokens = extraction_result.usage.get("prompt_tokens", 0)
                metrics.output_tokens = extraction_result.usage.get("output_tokens", 0)
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
                return self._process_individual(content, questions, config)

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
                        return_usage=True,
                        response_schema=config.response_schema,
                        **config.options,
                    )

                    extraction_result = self.response_extractor.extract_from_response(
                        response,
                        1,
                        config.response_schema,  # Single question
                    )

                    # For individual processing, we always get a single answer (str)
                    answer = (
                        extraction_result.answers
                        if not extraction_result.is_batch_result
                        else extraction_result.answers[0]
                    )
                    answers.append(answer)

                    metrics.prompt_tokens += extraction_result.usage.get(
                        "prompt_tokens", 0
                    )
                    metrics.output_tokens += extraction_result.usage.get(
                        "output_tokens", 0
                    )

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
            batch_prompt_tokens=batch_metrics.prompt_tokens,
            batch_output_tokens=batch_metrics.output_tokens,
            individual_time=individual_metrics.time,
            batch_time=batch_metrics.time,
        )

    def _process_with_response_processor(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        questions: List[str],
        config: ProcessingOptions,
    ) -> Dict[str, Any]:
        """Process questions with integrated ResponseProcessor"""
        try:
            with self._metrics_tracker(call_count=1) as batch_metrics:
                # Make batch API call
                raw_response = self.client.generate_batch(
                    content=content,
                    questions=questions,
                    return_usage=True,
                    response_schema=config.response_schema,
                    **config.options,
                )

                # Get result from ResponseProcessor
                result = config.response_processor.process_batch_response(
                    raw_response=raw_response,
                    questions=questions,
                    response_schema=config.response_schema,
                    return_usage=True,
                    api_call_time=batch_metrics.time,
                )

                # Add efficiency metrics if comparison requested
                if config.compare_methods:
                    individual_answers, individual_metrics = self._process_individual(
                        content, questions, config
                    )

                    # Extract batch metrics from ResponseProcessor result
                    batch_usage = result.get("usage", {})
                    batch_metrics.prompt_tokens = batch_usage.get("prompt_tokens", 0)
                    batch_metrics.output_tokens = batch_usage.get("output_tokens", 0)

                    self.result_builder.enhance_response_processor_result(
                        result, batch_metrics, individual_metrics, individual_answers
                    )

                return result

        except Exception as e:
            # Error handling with ResponseProcessor
            error_response = type(
                "ErrorResponse", (), {"text": f"Error: {str(e)}", "parsed": None}
            )()

            result = config.response_processor.process_batch_response(
                raw_response=error_response,
                questions=questions,
                response_schema=config.response_schema,
                return_usage=True,
            )
            result["processing_error"] = str(e)
            return result
