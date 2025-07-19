"""
Response processing module for Gemini Batch Framework

This module handles processing and validation of API responses from the Gemini API.
It provides structured output validation, quality assessment, and result packaging
for both individual and batch requests.
"""  # noqa: D212, D415

from .processor import ResponseProcessor
from .quality import calculate_quality_score
from .types import ExtractionResult, ProcessedResponse, ValidationResult

__all__ = [  # noqa: RUF022
    # Main processor
    "ResponseProcessor",
    # Core types
    "ExtractionResult",
    "ProcessedResponse",
    "ValidationResult",
    # Quality assessment
    "calculate_quality_score",
]
