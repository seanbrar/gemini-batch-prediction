"""
Response processing types and result containers

This module defines the core data structures used throughout the response
processing system, including extraction results, processed responses,
and validation outcomes.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class ExtractionResult:
    """Result from response extraction with usage metrics and quality data"""

    answers: Union[
        List[str], str
    ]  # List for batch (multiple questions), str for individual
    usage: Dict[str, int]
    structured_quality: Optional[Dict[str, Any]] = None

    @property
    def is_batch_result(self) -> bool:
        """True if this represents a batch response (multiple answers)"""
        return isinstance(self.answers, list)


@dataclass
class ProcessedResponse:
    """Complete processed response with metadata and quality metrics"""

    # Core results
    answers: List[str]
    success: bool

    # Metadata
    confidence: Optional[float] = None
    processing_method: str = "text_extraction"
    question_count: int = 0

    # Structured data (if applicable)
    structured_data: Optional[Any] = None
    schema_validation_success: bool = False

    # Quality metrics (if comparison available)
    quality_score: Optional[float] = None

    # Detailed metadata
    metadata: Dict[str, Any] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []

    @property
    def answer_count(self) -> int:
        """Number of answers extracted"""
        return len(self.answers)

    @property
    def has_structured_data(self) -> bool:
        """Whether structured data was successfully extracted and validated"""
        return self.structured_data is not None and self.schema_validation_success

    @property
    def has_quality_metrics(self) -> bool:
        """Whether quality comparison metrics are available"""
        return self.quality_score is not None


@dataclass
class ValidationResult:
    """Result from schema validation with confidence and error details"""

    success: bool
    parsed_data: Any
    confidence: float  # 0.0 to 1.0
    validation_method: str
    errors: List[str]
    raw_text: str


@dataclass
class ParsingResult:
    """Result from JSON parsing with method and error information"""

    success: bool
    parsed_data: Any
    confidence: float
    method: str
    errors: List[str]
