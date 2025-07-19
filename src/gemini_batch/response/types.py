"""
Response processing types and result containers

This module defines the core data structures used throughout the response
processing system, including extraction results, processed responses,
and validation outcomes.
"""

from dataclasses import dataclass, field
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
    """Result of structured response validation"""

    is_valid: bool
    corrected_data: Optional[Any] = None
    errors: Optional[str] = None
    quality_score: float = 0.0


@dataclass
class ProcessingMetrics:
    """Metrics for a single processing operation"""

    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    time: float = 0.0
    structured_output: Optional[Dict[str, Any]] = None

    @classmethod
    def empty(cls):
        return cls()

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def effective_tokens(self) -> int:
        """Tokens that are actually billed (uncached input + all output)"""
        return max(0, (self.prompt_tokens - self.cached_tokens) + self.output_tokens)

    @property
    def cache_hit_ratio(self) -> float:
        """Ratio of cached to total prompt tokens"""
        if self.prompt_tokens == 0:
            return 0.0
        return min(1.0, self.cached_tokens / self.prompt_tokens)  # Cap at 100%

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "effective_tokens": self.effective_tokens,
            "cache_hit_ratio": self.cache_hit_ratio,
            "time": self.time,
        }


@dataclass
class ProcessingOptions:
    """Options for processing a request"""

    compare_methods: bool = False
    return_usage: bool = False
    response_schema: Optional[Any] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsingResult:
    """Result from JSON parsing with method and error information"""

    success: bool
    parsed_data: Any
    confidence: float
    method: str
    errors: List[str]
