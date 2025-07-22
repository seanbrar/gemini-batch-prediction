"""Response processing types and result containers

This module defines the core data structures used throughout the response
processing system, including extraction results, processed responses,
and validation outcomes.
"""  # noqa: D415

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractionResult:
    """Result from response extraction with usage metrics and quality data"""  # noqa: D415

    answers: list[str] | str  # List for batch (multiple questions), str for individual
    usage: dict[str, int]
    structured_quality: dict[str, Any] | None = None

    @property
    def is_batch_result(self) -> bool:
        """True if this represents a batch response (multiple answers)"""  # noqa: D415
        return isinstance(self.answers, list)


@dataclass
class ProcessedResponse:
    """Complete processed response with metadata and quality metrics"""  # noqa: D415

    # Core results
    answers: list[str]
    success: bool

    # Metadata
    confidence: float | None = None
    processing_method: str = "text_extraction"
    question_count: int = 0

    # Structured data (if applicable)
    structured_data: Any | None = None
    schema_validation_success: bool = False

    # Quality metrics (if comparison available)
    quality_score: float | None = None

    # Detailed metadata
    metadata: dict[str, Any] = None
    errors: list[str] = None

    def __post_init__(self):  # noqa: D105
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []

    @property
    def answer_count(self) -> int:
        """Number of answers extracted"""  # noqa: D415
        return len(self.answers)

    @property
    def has_structured_data(self) -> bool:
        """Whether structured data was successfully extracted and validated"""  # noqa: D415
        return self.structured_data is not None and self.schema_validation_success

    @property
    def has_quality_metrics(self) -> bool:
        """Whether quality comparison metrics are available"""  # noqa: D415
        return self.quality_score is not None


@dataclass
class ValidationResult:
    """Result of structured response validation"""  # noqa: D415

    is_valid: bool
    corrected_data: Any | None = None
    errors: str | None = None
    quality_score: float = 0.0


@dataclass
class ProcessingMetrics:
    """Metrics for a single processing operation"""  # noqa: D415

    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    time: float = 0.0
    structured_output: dict[str, Any] | None = None

    @classmethod
    def empty(cls):  # noqa: D102
        return cls()

    @property
    def total_tokens(self) -> int:  # noqa: D102
        return self.prompt_tokens + self.output_tokens

    @property
    def effective_tokens(self) -> int:
        """Tokens that are actually billed (uncached input + all output)"""  # noqa: D415
        return max(0, (self.prompt_tokens - self.cached_tokens) + self.output_tokens)

    @property
    def cache_hit_ratio(self) -> float:
        """Ratio of cached to total prompt tokens"""  # noqa: D415
        if self.prompt_tokens == 0:
            return 0.0
        return min(1.0, self.cached_tokens / self.prompt_tokens)  # Cap at 100%

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
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
    """Options for processing a request"""  # noqa: D415

    compare_methods: bool = False
    return_usage: bool = False
    response_schema: Any | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsingResult:
    """Result from JSON parsing with method and error information"""  # noqa: D415

    success: bool
    parsed_data: Any
    confidence: float
    method: str
    errors: list[str]
