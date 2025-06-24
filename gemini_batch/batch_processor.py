"""
Batch processor for text content analysis
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from .client import GeminiClient
from .efficiency import track_efficiency
from .response import extract_answers


class BatchProcessor:
    """Process multiple questions efficiently using batch operations"""

    def __init__(self, client: GeminiClient = None, **client_kwargs):
        """Initialize with unified client"""
        if client is not None:
            self.client = client
        else:
            self.client = GeminiClient(**client_kwargs)

        self.reset_metrics()

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

        # Auto-create ResponseProcessor when response_schema is provided but no processor
        if response_schema is not None and response_processor is None:
            from .response import ResponseProcessor

            response_processor = ResponseProcessor()

        # If response_processor is provided, delegate all response processing to it
        if response_processor is not None:
            return self._process_with_response_processor(
                content,
                questions,
                response_processor,
                response_schema,
                compare_methods,
                **options,
            )

        # Standard processing path (existing logic)
        self.reset_metrics()

        # Process batch first (unified for all content types)
        batch_answers, batch_metrics = self._process_batch(
            content, questions, return_usage, response_schema, **options
        )

        # Optionally process individually for comparison
        if compare_methods:
            individual_answers, individual_metrics = self._process_individual(
                content, questions, return_usage, response_schema, **options
            )
        else:
            individual_answers = None
            individual_metrics = {
                "calls": 0,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "tokens": 0,
                "time": 0.0,
            }

        # Calculate efficiency metrics
        efficiency = track_efficiency(
            individual_calls=individual_metrics["calls"],
            batch_calls=batch_metrics["calls"],
            individual_prompt_tokens=individual_metrics.get("prompt_tokens", 0),
            individual_output_tokens=individual_metrics.get("output_tokens", 0),
            batch_prompt_tokens=batch_metrics.get("prompt_tokens", 0),
            batch_output_tokens=batch_metrics.get("output_tokens", 0),
            individual_time=individual_metrics["time"],
            batch_time=batch_metrics["time"],
        )

        result = {
            "question_count": len(questions),
            "answers": batch_answers,
            "efficiency": efficiency,
            "metrics": {
                "batch": batch_metrics,
                "individual": individual_metrics,
            },
        }

        # Add structured data if available for easy access
        if batch_metrics.get("structured_output", {}).get("structured_data"):
            result["structured_data"] = batch_metrics["structured_output"][
                "structured_data"
            ]

        if compare_methods and individual_answers:
            result["individual_answers"] = individual_answers

        if return_usage:
            result["usage"] = {
                "prompt_tokens": batch_metrics.get("prompt_tokens", 0),
                "output_tokens": batch_metrics.get("output_tokens", 0),
                "total_tokens": batch_metrics.get("tokens", 0),
                "cached_tokens": 0,
            }

        return result

    def _process_batch(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        questions: List[str],
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> tuple[List[str], Dict[str, Any]]:
        """Process all questions in a single batch call"""
        start_time = time.time()

        try:
            # Single unified call handles any content type
            response = self.client.generate_batch(
                content=content,
                questions=questions,
                return_usage=True,
                response_schema=response_schema,
                **options,
            )

            # Extract answers and metrics based on response type
            if isinstance(response, dict):
                # Response includes usage and possibly parsed data
                usage = response.get(
                    "usage", {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                )

                structured_quality = None  # Initialize for all paths

                if response_schema:
                    if (
                        "structured_success" in response
                        and response["structured_success"]
                    ):
                        # High-quality structured output - preserve structured data
                        parsed_data = response.get("parsed")

                        # For structured output, provide minimal text representation for
                        # compatibility but also preserve the actual structured object
                        answers = (
                            [str(parsed_data)]
                            if parsed_data
                            else ["No structured data"]
                        )

                        # Add structured output metadata with actual structured object
                        structured_quality = {
                            "confidence": response.get("structured_confidence", 0.0),
                            "method": response.get("validation_method", "unknown"),
                            "errors": response.get("validation_errors", []),
                            "structured_data": parsed_data,
                        }
                    else:
                        # Fallback to text extraction with lower confidence
                        response_text = response.get("text", "")
                        answers = extract_answers(
                            response_text, len(questions), is_structured=False
                        )
                        structured_quality = {
                            "confidence": 0.3,  # Lower confidence for text fallback
                            "method": "text_fallback",
                            "errors": response.get(
                                "validation_errors", ["Structured parsing failed"]
                            ),
                        }
                else:
                    # Standard text response
                    response_text = response.get("text", "")
                    answers = extract_answers(
                        response_text, len(questions), is_structured=False
                    )
            else:
                # Simple text response (no usage tracking)
                usage = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                answers = extract_answers(response, len(questions), is_structured=False)
                structured_quality = None

            duration = time.time() - start_time

            metrics = {
                "calls": 1,
                "prompt_tokens": usage["prompt_tokens"],
                "output_tokens": usage["output_tokens"],
                "tokens": usage["total_tokens"],
                "time": duration,
            }

            # Add structured output quality metrics if applicable
            if response_schema and structured_quality:
                metrics["structured_output"] = structured_quality

            return answers, metrics

        except Exception:
            # Fallback to individual processing
            return self._process_individual(
                content, questions, return_usage, response_schema, **options
            )

    def _process_individual(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        questions: List[str],
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> tuple[List[str], Dict[str, Any]]:
        """Process questions individually for comparison or fallback"""
        answers = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        start_time = time.time()

        for question in questions:
            try:
                # Single unified call handles any content type
                response = self.client.generate_content(
                    content=content,
                    prompt=question,
                    return_usage=True,
                    response_schema=response_schema,
                    **options,
                )

                if isinstance(response, dict):
                    if response_schema and "parsed" in response:
                        # Structured output - convert parsed object to string
                        parsed_data = response["parsed"]
                        if parsed_data is not None:
                            answers.append(str(parsed_data))
                        else:
                            answers.append(response.get("text", ""))
                    else:
                        # Text response
                        answers.append(response.get("text", ""))

                    usage = response.get(
                        "usage", {"prompt_tokens": 0, "output_tokens": 0}
                    )
                    total_prompt_tokens += usage["prompt_tokens"]
                    total_output_tokens += usage["output_tokens"]
                else:
                    # Simple response - convert to string if needed
                    answers.append(str(response) if response is not None else "")

            except Exception as e:
                answers.append(f"Error: {str(e)}")

        duration = time.time() - start_time

        metrics = {
            "calls": len(questions),
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": total_output_tokens,
            "tokens": total_prompt_tokens + total_output_tokens,
            "time": duration,
        }

        return answers, metrics

    def process_questions_multi_source(
        self,
        sources: List[Union[str, Path, List[Union[str, Path]]]],
        questions: List[str],
        response_schema: Optional[Any] = None,
        response_processor=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process multiple sources in a single batch for maximum efficiency

        Combines multiple distinct sources (files, URLs, text, directories) and
        processes all questions against all sources simultaneously.

        See docs/SOURCE_HANDLING.md for detailed usage examples.
        """
        if not sources:
            raise ValueError("At least one source is required")
        if not questions:
            raise ValueError("Questions are required")

        # Flatten the sources list to create a single combined content input
        # The unified client can handle mixed content types in a single call
        combined_content = sources

        # Use the main process_questions method (which now handles response_processor)
        result = self.process_questions(
            combined_content,
            questions,
            response_schema=response_schema,
            response_processor=response_processor,
            **kwargs,
        )

        # Enhance result with multi-source metadata
        result["source_count"] = len(sources)
        result["processing_mode"] = "multi_source_batch"

        return result

    def _process_with_response_processor(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        questions: List[str],
        response_processor,
        response_schema: Optional[Any] = None,
        compare_methods: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process questions with integrated ResponseProcessor for unified handling

        This method provides clean separation of concerns:
        - BatchProcessor handles content processing and API calls
        - ResponseProcessor handles all response processing and result packaging

        Args:
            content: Content to process (any supported type)
            questions: List of questions
            response_processor: ResponseProcessor instance (required for integration)
            response_schema: Optional schema for structured output
            compare_methods: Whether to run both batch and individual for comparison
            **kwargs: Additional options

        Returns:
            Complete result dict from ResponseProcessor, enhanced with efficiency metrics if compare_methods=True
        """

        try:
            # Measure total time including API call
            total_start_time = time.time()

            # Always run batch processing
            raw_response = self.client.generate_batch(
                content=content,
                questions=questions,
                return_usage=True,
                response_schema=response_schema,
                **kwargs,
            )

            # Calculate total time including API call
            total_duration = time.time() - total_start_time

            # Get the primary result from ResponseProcessor with API call timing
            result = response_processor.process_batch_response(
                raw_response=raw_response,
                questions=questions,
                response_schema=response_schema,
                return_usage=True,
                api_call_time=total_duration,
            )

            # If comparison is requested, run individual processing and add efficiency metrics
            if compare_methods:
                # Extract batch metrics from the ResponseProcessor result
                batch_usage = result.get("usage", {})
                batch_time = result.get("processing_time", 0.0)

                batch_metrics = {
                    "calls": 1,
                    "prompt_tokens": batch_usage.get("prompt_tokens", 0),
                    "output_tokens": batch_usage.get("output_tokens", 0),
                    "tokens": batch_usage.get("total_tokens", 0),
                    "time": batch_time,
                }

                # Run individual processing for comparison
                individual_answers, individual_metrics = self._process_individual(
                    content,
                    questions,
                    return_usage=True,
                    response_schema=response_schema,
                    **kwargs,
                )

                # Calculate efficiency metrics
                efficiency = track_efficiency(
                    individual_calls=individual_metrics["calls"],
                    batch_calls=batch_metrics["calls"],
                    individual_prompt_tokens=individual_metrics.get("prompt_tokens", 0),
                    individual_output_tokens=individual_metrics.get("output_tokens", 0),
                    batch_prompt_tokens=batch_metrics.get("prompt_tokens", 0),
                    batch_output_tokens=batch_metrics.get("output_tokens", 0),
                    individual_time=individual_metrics["time"],
                    batch_time=batch_metrics["time"],
                )

                # Add efficiency data to the result
                result["efficiency"] = efficiency
                result["metrics"] = {
                    "batch": batch_metrics,
                    "individual": individual_metrics,
                }
                result["individual_answers"] = individual_answers

            return result

        except Exception as e:
            error_response = type(
                "ErrorResponse", (), {"text": f"Error: {str(e)}", "parsed": None}
            )()

            result = response_processor.process_batch_response(
                raw_response=error_response,
                questions=questions,
                response_schema=response_schema,
                return_usage=True,
            )
            result["processing_error"] = str(e)
            return result

    def reset_metrics(self):
        """Reset tracking metrics"""
        pass  # Simplified - metrics now handled per-call
