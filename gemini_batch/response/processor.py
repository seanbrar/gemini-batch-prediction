"""
Response processor for structured JSON outputs from the Gemini API

This module handles the extraction, validation, and packaging of responses
from the Gemini API. It supports both structured JSON responses and fallback
text parsing, with built-in quality assessment and error handling.
"""

import json
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from .quality import calculate_quality_score
from .types import ExtractionResult, ProcessedResponse


class ResponseProcessor:
    """Processes and validates responses from the Gemini API"""

    def __init__(self):
        """Initialize the response processor"""
        pass

    def process_batch_response(
        self,
        raw_response,
        questions: List[str],
        response_schema: Optional[Any] = None,
        return_usage: bool = True,
        comparison_answers: Optional[List[str]] = None,
        api_call_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Process a raw batch API response with comprehensive result packaging

        This is the primary integration method that BatchProcessor should use.
        It handles all response processing, packaging, and result formatting.

        Args:
            raw_response: Raw API response from batch call
            questions: List of questions that were asked
            response_schema: Optional schema for structured output validation
            return_usage: Whether to include usage metrics
            comparison_answers: Optional answers for quality comparison
            api_call_time: Total time for API call (if provided, used instead of processing time)

        Returns:
            Complete result dict with processed response and all metadata
        """
        import time

        start_time = time.time()

        # Process the response using simplified logic
        processed_response = self.process_response(
            response=raw_response,
            expected_questions=len(questions),
            schema=response_schema,
            comparison_answers=comparison_answers,
            return_confidence=True,
        )

        # Extract usage information from raw response
        usage = {}
        if return_usage:
            if isinstance(raw_response, dict):
                usage = raw_response.get(
                    "usage", {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                )
            # For non-dict responses, usage extraction could be enhanced here

        # Use API call time if provided, otherwise use processing time
        processing_time = (
            api_call_time if api_call_time is not None else (time.time() - start_time)
        )

        # Package complete result
        return {
            "processed_response": processed_response,
            "raw_response": raw_response,
            "answers": processed_response.answers,
            "usage": usage,
            "success": processed_response.success,
            "processing_method": processed_response.processing_method,
            "confidence": processed_response.confidence,
            "structured_data": processed_response.structured_data,
            "processing_time": processing_time,
            "has_structured_data": processed_response.has_structured_data,
            "question_count": len(questions),
            "schema_provided": response_schema is not None,
        }

    def process_response(
        self,
        response,
        expected_questions: int,
        schema: Optional[Any] = None,
        comparison_answers: Optional[List[str]] = None,
        return_confidence: bool = True,
    ) -> ProcessedResponse:
        """
        Process an API response with structured validation and quality assessment

        Args:
            response: The API response object
            expected_questions: Number of questions/answers expected
            schema: Optional schema for structured validation
            comparison_answers: Optional answers for quality comparison
            return_confidence: Whether to calculate confidence metrics

        Returns:
            ProcessedResponse with all relevant data and metadata
        """
        # Extract answers using simplified logic
        extraction_result = self.extract_answers_from_response(
            response, expected_questions, schema
        )

        # Determine success based on errors
        success = (
            not extraction_result.structured_quality
            or not extraction_result.structured_quality.get("errors")
        )

        # Calculate confidence
        confidence = None
        if return_confidence:
            if extraction_result.structured_quality:
                confidence = extraction_result.structured_quality.get("confidence", 0.5)
            else:
                confidence = 0.8 if success else 0.3

        # Calculate quality score if comparison available
        quality_score = None
        if comparison_answers and extraction_result.answers:
            if isinstance(extraction_result.answers, list):
                quality_score = calculate_quality_score(
                    comparison_answers, extraction_result.answers
                )

        return ProcessedResponse(
            answers=extraction_result.answers
            if isinstance(extraction_result.answers, list)
            else [extraction_result.answers],
            success=success,
            confidence=confidence,
            processing_method="simplified_structured",
            question_count=expected_questions,
            structured_data=extraction_result.structured_quality.get("structured_data")
            if extraction_result.structured_quality
            else None,
            schema_validation_success=success,
            quality_score=quality_score,
            metadata={
                "method": "simplified_structured",
                "structured": schema is not None,
            },
            errors=extraction_result.structured_quality.get("errors", [])
            if extraction_result.structured_quality
            else [],
        )

    def extract_answers_from_response(
        self, response: Any, question_count: int, response_schema: Optional[Any] = None
    ) -> ExtractionResult:
        """
        Extract answers from API response using structured validation with fallback parsing

        Implements a "Trust, but Verify" approach: first attempts to use structured
        data from response.parsed, then falls back to parsing response.text as JSON
        if needed. Validates against provided schema when available.
        """
        # Extract usage metrics with enhanced cache awareness
        usage = {
            "prompt_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
        }  # Default with cache

        if isinstance(response, dict):
            # Enhanced usage extraction
            response_usage = response.get("usage", {})
            usage.update(
                {
                    "prompt_tokens": response_usage.get("prompt_tokens", 0),
                    "output_tokens": response_usage.get("output_tokens", 0),
                    "cached_tokens": response_usage.get("cached_tokens", 0),
                    "cache_hit_ratio": response_usage.get("cache_hit_ratio", 0.0),
                    "cache_enabled": response_usage.get("cache_enabled", False),
                }
            )
        else:
            # Extract from response object using enhanced metrics extraction
            from ..efficiency.metrics import extract_usage_metrics

            usage = extract_usage_metrics(response)

        # Auto-detect batch vs individual
        is_batch = question_count > 1
        structured_quality = None
        parsed_data = None
        answers = []
        errors = []

        # Path A: Use structured data from response.parsed if available
        if hasattr(response, "parsed") and response.parsed:
            parsed_data = response.parsed
        else:
            # Path B: Parse response.text as JSON with validation
            try:
                response_text = getattr(response, "text", "")
                if response_text:
                    loaded_json = json.loads(response_text)
                    if response_schema:
                        # Validate the JSON against the provided schema
                        try:
                            parsed_data = response_schema.model_validate(loaded_json)
                        except ValidationError as e:
                            errors.append(
                                f"Response JSON did not match the provided schema: {e}"
                            )
                    else:
                        # Use as default response format (list of strings)
                        parsed_data = loaded_json
                else:
                    errors.append("Response was empty.")
            except json.JSONDecodeError as e:
                errors.append(f"Failed to decode JSON from response text: {e}")
            except Exception as e:
                errors.append(f"An unexpected error occurred during JSON parsing: {e}")

        # Convert final parsed data to answer list
        if parsed_data:
            if response_schema:
                # For custom schemas, the whole object is the "answer"
                answers = [str(parsed_data)]
            elif isinstance(parsed_data, list):
                # Default case: a list of string answers
                answers = [str(item) for item in parsed_data]
        else:
            # If all parsing and validation fails
            answers = [f"Error: Could not parse response. Details: {errors}"]

        # Handle individual vs batch response format
        if not is_batch and len(answers) == 1:
            # For individual questions, return the single answer as a string
            final_answers = answers[0]
        else:
            # For batch questions, return the list of answers
            final_answers = answers

        # Set up structured quality data if we have a schema
        if response_schema:
            structured_quality = {
                "confidence": 0.9
                if not errors
                else 0.3,  # High confidence if no errors
                "method": "structured_validation" if not errors else "fallback_parsing",
                "errors": errors,
                "structured_data": parsed_data,
            }

        return ExtractionResult(
            answers=final_answers,
            usage=usage,
            structured_quality=structured_quality,
        )

    # Convenience methods for specific use cases

    def extract_structured_data(
        self, response, schema: Any, return_confidence: bool = True
    ) -> ProcessedResponse:
        """Extract and validate structured data from response using provided schema"""
        return self.process_response(
            response=response,
            expected_questions=0,
            schema=schema,
            return_confidence=return_confidence,
        )

    def extract_text_answers(
        self,
        response,
        question_count: int,
        comparison_answers: Optional[List[str]] = None,
    ) -> ProcessedResponse:
        """Extract text answers without structured schema validation"""
        return self.process_response(
            response=response,
            expected_questions=question_count,
            schema=None,
            comparison_answers=comparison_answers,
        )
