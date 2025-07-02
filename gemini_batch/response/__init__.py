"""
Response processing module for Gemini Batch Framework
"""

from .extraction import extract_answers, extract_answers_from_text
from .parsing import (
    is_float,
    parse_text_with_schema_awareness,
    try_enhanced_pattern_parsing,
    try_json_extraction,
)
from .processor import ResponseProcessor
from .quality import calculate_quality_score
from .types import ExtractionResult, ParsingResult, ProcessedResponse, ValidationResult
from .validation import (
    validate_against_schema,
    validate_generic_type,
    validate_structured_response,
)

__all__ = [
    # Central interface
    "ResponseProcessor",
    "ProcessedResponse",
    # Result types
    "ExtractionResult",
    "ValidationResult",
    "ParsingResult",
    # Individual components
    "extract_answers",
    "extract_answers_from_text",
    "validate_against_schema",
    "validate_generic_type",
    "validate_structured_response",
    "is_float",
    "parse_text_with_schema_awareness",
    "try_enhanced_pattern_parsing",
    "try_json_extraction",
    "calculate_quality_score",
]
