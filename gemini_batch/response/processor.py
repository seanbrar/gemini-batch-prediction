"""
Central response processor that coordinates validation, parsing, extraction, and quality assessment
"""

from typing import Any, Dict, List, Optional

from .extraction import extract_answers
from .parsing import parse_text_with_schema_awareness
from .quality import calculate_quality_score
from .types import ProcessedResponse
from .validation import validate_against_schema, validate_structured_response


class ResponseProcessor:
    """Central processor for handling all types of response processing"""

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
        Process a raw batch API response with complete integration handling

        This is the primary integration method that BatchProcessor should use.
        It handles all response processing, packaging, and result formatting.

        Args:
            raw_response: Raw API response from batch call
            questions: List of questions that were asked
            response_schema: Optional schema for structured output
            return_usage: Whether to include usage metrics
            comparison_answers: Optional answers for quality comparison
            api_call_time: Total time for API call (if provided, used instead of processing time)

        Returns:
            Complete result dict with processed response and all metadata
        """
        import time

        start_time = time.time()

        # Process the response using existing logic
        processed_response = self.process_response(
            response=raw_response,
            expected_questions=len(questions),
            schema=response_schema,
            is_structured=response_schema is not None,
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
        is_structured: bool = None,
        comparison_answers: Optional[List[str]] = None,
        return_confidence: bool = True,
    ) -> ProcessedResponse:
        """
        Process an API response with unified handling for both structured and unstructured responses

        Args:
            response: The API response object
            expected_questions: Number of questions/answers expected
            schema: Optional schema for structured validation
            is_structured: Whether to treat as structured response (auto-detected if None)
            comparison_answers: Optional answers for quality comparison
            return_confidence: Whether to calculate confidence metrics

        Returns:
            ProcessedResponse with all relevant data and metadata
        """

        # Auto-detect structured response if not specified
        if is_structured is None:
            is_structured = schema is not None or (
                hasattr(response, "parsed") and response.parsed is not None
            )

        # Process based on type
        if is_structured and schema is not None:
            return self._process_structured_response(
                response,
                expected_questions,
                schema,
                comparison_answers,
                return_confidence,
            )
        else:
            return self._process_text_response(
                response, expected_questions, comparison_answers, return_confidence
            )

    def _process_structured_response(
        self,
        response,
        expected_questions: int,
        schema: Any,
        comparison_answers: Optional[List[str]],
        return_confidence: bool,
    ) -> ProcessedResponse:
        """Process a structured response with schema validation"""

        # Step 1: Try direct validation of response.parsed
        validation_result = validate_structured_response(response, schema)

        if validation_result.success:
            # Direct validation succeeded
            response_data = {
                "parsed": validation_result.parsed_data,
                "text": validation_result.raw_text,
            }
            answers = extract_answers(
                response_data, expected_questions, is_structured=True
            )

            return ProcessedResponse(
                answers=answers,
                success=True,
                confidence=validation_result.confidence if return_confidence else None,
                processing_method=validation_result.validation_method,
                question_count=expected_questions,
                structured_data=validation_result.parsed_data,
                schema_validation_success=True,
                quality_score=self._calculate_quality_if_needed(
                    answers, comparison_answers
                ),
                metadata={
                    "validation_method": validation_result.validation_method,
                    "raw_text_length": len(validation_result.raw_text),
                    "structured": True,
                },
                errors=validation_result.errors,
            )
        else:
            # Direct validation failed - try parsing text then validating
            return self._fallback_parse_then_validate(
                response,
                expected_questions,
                schema,
                comparison_answers,
                return_confidence,
            )

    def _fallback_parse_then_validate(
        self,
        response,
        expected_questions: int,
        schema: Any,
        comparison_answers: Optional[List[str]],
        return_confidence: bool,
    ) -> ProcessedResponse:
        """Fallback: parse text then validate the result"""

        raw_text = getattr(response, "text", str(response))

        # Step 1: Parse text
        parsing_result = parse_text_with_schema_awareness(raw_text, schema)

        if parsing_result.success:
            # Step 2: Validate parsed result
            try:
                validated_data = validate_against_schema(
                    parsing_result.parsed_data, schema
                )

                # Create structured response
                response_data = {"parsed": validated_data, "text": raw_text}
                answers = extract_answers(
                    response_data, expected_questions, is_structured=True
                )

                return ProcessedResponse(
                    answers=answers,
                    success=True,
                    confidence=parsing_result.confidence if return_confidence else None,
                    processing_method=f"parse_then_validate_{parsing_result.method}",
                    question_count=expected_questions,
                    structured_data=validated_data,
                    schema_validation_success=True,
                    quality_score=self._calculate_quality_if_needed(
                        answers, comparison_answers
                    ),
                    metadata={
                        "parsing_method": parsing_result.method,
                        "raw_text_length": len(raw_text),
                        "structured": True,
                    },
                    errors=[],
                )

            except Exception:
                # Validation failed - fall back to text processing
                return self._process_text_response(
                    response, expected_questions, comparison_answers, return_confidence
                )
        else:
            # Parsing failed - fall back to text processing
            return self._process_text_response(
                response, expected_questions, comparison_answers, return_confidence
            )

    def _process_text_response(
        self,
        response,
        expected_questions: int,
        comparison_answers: Optional[List[str]],
        return_confidence: bool,
    ) -> ProcessedResponse:
        """Process a text-based response"""

        # Extract answers using text extraction
        answers = extract_answers(response, expected_questions, is_structured=False)

        # Basic success check - did we get reasonable answers?
        success = len(answers) > 0 and not all(
            "No answer found" in answer for answer in answers
        )

        return ProcessedResponse(
            answers=answers,
            success=success,
            confidence=0.8 if success else 0.3,  # Basic confidence estimate
            processing_method="text_extraction",
            question_count=expected_questions,
            structured_data=None,
            schema_validation_success=False,
            quality_score=self._calculate_quality_if_needed(
                answers, comparison_answers
            ),
            metadata={
                "text_length": len(str(response)),
                "structured": False,
            },
            errors=[],
        )

    def _calculate_quality_if_needed(
        self, answers: List[str], comparison_answers: Optional[List[str]]
    ) -> Optional[float]:
        """Calculate quality score if comparison answers are provided"""
        if comparison_answers is not None:
            return calculate_quality_score(comparison_answers, answers)
        return None

    # Convenience methods for specific use cases

    def extract_structured_data(
        self, response, schema: Any, return_confidence: bool = True
    ) -> ProcessedResponse:
        """Extract structured data without answer extraction"""
        return self.process_response(
            response=response,
            expected_questions=0,
            schema=schema,
            is_structured=True,
            return_confidence=return_confidence,
        )

    def extract_text_answers(
        self,
        response,
        question_count: int,
        comparison_answers: Optional[List[str]] = None,
    ) -> ProcessedResponse:
        """Extract text answers without structured processing"""
        return self.process_response(
            response=response,
            expected_questions=question_count,
            schema=None,
            is_structured=False,
            comparison_answers=comparison_answers,
        )
