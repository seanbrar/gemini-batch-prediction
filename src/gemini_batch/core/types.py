"""Core data types that flow through the pipeline.

This module defines the immutable data structures that represent the state
of a request as it moves through different processing stages. Each stage
transforms the data into a new state, ensuring type safety and preventing
invalid state transitions.
"""

from __future__ import annotations

import dataclasses
import typing

# Note: This import is for type-checking/static analysis only and ensures
# external dependency accounting tests detect the SDK dependency without
# introducing a runtime import in core modules.
if typing.TYPE_CHECKING:  # pragma: no cover - import for dependency visibility only
    from collections.abc import Callable
    from pathlib import Path

    # Ensure external dependency visibility for tests without runtime import
    from google.genai import types as genai_types  # noqa: F401

    from gemini_batch.config import FrozenConfig, GeminiConfig

# --- Result Monad for Robust Error Handling ---
# Implements the "Unified Result Type" enhancement for explicit error handling.
# This prevents the need for broad try/except blocks in the executor and makes
# failures a predictable part of the data flow.

TSuccess = typing.TypeVar("TSuccess")
TFailure = typing.TypeVar("TFailure", bound=Exception)


@dataclasses.dataclass(frozen=True)
class Success[TSuccess]:
    """A successful result in the pipeline."""

    value: TSuccess


@dataclasses.dataclass(frozen=True)
class Failure[TFailure]:
    """A failure in the pipeline, containing the error."""

    error: TFailure


Result = Success[TSuccess] | Failure[TFailure]

# --- Core Data Models ---


@dataclasses.dataclass(frozen=True)
class ConversationTurn:
    """A single turn in a conversation history."""

    question: str
    answer: str
    is_error: bool = False


@dataclasses.dataclass(frozen=True)
class Source:
    """A structured representation of a single input source.

    Content access is lazy via the `content_loader` callable to optimize
    memory usage, ensuring content is only loaded when needed.
    """

    source_type: typing.Literal["text", "youtube", "arxiv", "file"]
    identifier: str | Path  # The original path, URL, or text identifier
    mime_type: str
    size_bytes: int
    content_loader: Callable[[], bytes]  # A function to get content on demand


# --- Typed Command States ---
# These dataclasses define the shape of our data as it is transformed by
# each stage of the pipeline.


@dataclasses.dataclass(frozen=True)
class TokenEstimate:
    """Range-based token estimate with confidence.

    Models uncertainty explicitly through ranges, allowing conservative
    or optimistic decisions based on use case needs.
    """

    min_tokens: int
    expected_tokens: int
    max_tokens: int
    confidence: float
    breakdown: dict[str, TokenEstimate] | None = None

    def __post_init__(self) -> None:
        """Validate invariants for ordering and bounds."""
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be >= 0, got {self.min_tokens}")
        if self.max_tokens < self.min_tokens:
            raise ValueError(
                f"max_tokens must be >= min_tokens, got min_tokens={self.min_tokens}, max_tokens={self.max_tokens}"
            )
        if self.expected_tokens < 0:
            raise ValueError(
                f"expected_tokens must be >= 0, got {self.expected_tokens}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be within [0.0, 1.0], got {self.confidence}"
            )


@dataclasses.dataclass(frozen=True)
class InitialCommand:
    """The initial state of a request, created by the user."""

    sources: tuple[typing.Any, ...]
    prompts: tuple[str, ...]
    config: FrozenConfig | GeminiConfig  # Support both during migration
    history: tuple[ConversationTurn, ...] = dataclasses.field(default_factory=tuple)


@dataclasses.dataclass(frozen=True)
class ResolvedCommand:
    """The state after sources have been resolved."""

    initial: InitialCommand
    resolved_sources: tuple[Source, ...]


# --- Library-owned neutral API payload types ---


@dataclasses.dataclass(frozen=True)
class TextPart:
    """A minimal library-owned representation of a text part.

    Additional part types (e.g., images, files) can be added later while
    keeping the core decoupled from any vendor SDK types.
    """

    text: str


# Neutral union of API parts; extendable without leaking provider types
type APIPart = TextPart | "FileRefPart" | "FilePlaceholder"


@dataclasses.dataclass(frozen=True)
class FileRefPart:
    """Provider-agnostic reference to uploaded content."""

    uri: str
    mime_type: str | None = None


@dataclasses.dataclass(frozen=True)
class FilePlaceholder:
    """Placeholder for a local file intended to be uploaded by the provider.

    Handlers replace this with a `FileRefPart` when uploads are supported.
    """

    local_path: Path
    mime_type: str | None = None


@dataclasses.dataclass(frozen=True)
class RateConstraint:
    """Immutable rate limit specification.

    All rates are per-minute. Tokens-per-minute may be omitted.
    """

    requests_per_minute: int
    tokens_per_minute: int | None = None
    min_interval_ms: int = 0
    burst_factor: float = 1.0

    def __post_init__(self) -> None:
        """Clamp provided values to safe minimums."""
        if self.requests_per_minute <= 0:
            object.__setattr__(self, "requests_per_minute", 1)
        if self.tokens_per_minute is not None and self.tokens_per_minute <= 0:
            object.__setattr__(self, "tokens_per_minute", None)
        if self.burst_factor < 1.0:
            object.__setattr__(self, "burst_factor", 1.0)


# For the minimal slice, a generation config is simply a mapping. We keep it
# library-owned and translate to provider-specific types at the API seam.
GenerationConfigDict = dict[str, object]


@dataclasses.dataclass(frozen=True)
class APICall:
    """A description of a single API call to be made."""

    model_name: str
    api_parts: tuple[APIPart, ...]
    api_config: GenerationConfigDict
    cache_name_to_use: str | None = None


@dataclasses.dataclass(frozen=True)
class ExecutionPlan:
    """Instructions for executing API calls.

    This includes both the primary call and an optional fallback call
    for when the primary call fails.
    """

    primary_call: APICall
    fallback_call: APICall | None = None  # For when batching fails
    # Optional rate limiting constraint
    rate_constraint: RateConstraint | None = None
    # Optional pre-generation actions
    upload_tasks: tuple[UploadTask, ...] = ()
    explicit_cache: ExplicitCachePlan | None = None


@dataclasses.dataclass(frozen=True)
class UploadTask:
    """Instruction to upload a local file and substitute an API part."""

    part_index: int
    local_path: Path
    mime_type: str | None = None
    required: bool = True


@dataclasses.dataclass(frozen=True)
class ExplicitCachePlan:
    """Instruction to create/use explicit cached content for context."""

    create: bool = False
    cache_name: str | None = None
    contents_part_indexes: tuple[int, ...] = ()
    # Whether to include system instruction text in the cached contents
    include_system_instruction: bool = True
    # Desired TTL in seconds for the cached content (provider may clamp/ignore)
    ttl_seconds: int | None = None
    # Deterministic key used to look up/store provider cache name in registry
    deterministic_key: str | None = None


@dataclasses.dataclass(frozen=True)
class PlannedCommand:
    """The state after an execution plan has been created."""

    resolved: ResolvedCommand
    execution_plan: ExecutionPlan
    token_estimate: TokenEstimate | None = None


@dataclasses.dataclass(frozen=True)
class FinalizedCommand:
    """The state after API calls have been executed."""

    planned: PlannedCommand
    raw_api_response: typing.Any
    # This will be populated by a future Telemetry handler/context.
    # Note: This field is intentionally mutable to collect metrics and is not
    # part of the immutability guarantees of the surrounding dataclass.
    telemetry_data: dict[str, object] = dataclasses.field(default_factory=dict)


# --- Result Types ---
# User-facing result structures returned by the pipeline.


class ResultEnvelope(typing.TypedDict, total=False):
    """Stable result shape for all extractions.

    This type ensures consistent structure regardless of extraction method.
    The 'total=False' allows optional fields while maintaining type safety.

    This is the main result structure that users receive from the pipeline.
    """

    # Core fields (always present)
    success: bool  # Always True (extraction never fails)
    answers: list[str]  # Always present, padded if needed
    extraction_method: str  # Which transform/fallback succeeded
    confidence: float  # 0.0-1.0 extraction confidence

    # Optional fields
    structured_data: typing.Any  # Original structured data if available
    metrics: dict[str, typing.Any]  # Telemetry metrics
    usage: dict[str, typing.Any]  # Token usage data
    diagnostics: dict[str, typing.Any]  # When diagnostics enabled
    validation_warnings: tuple[str, ...]  # Schema/contract violations
