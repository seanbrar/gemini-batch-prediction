"""Core data types that flow through the pipeline.

This module defines the immutable data structures that represent the state
of a request as it moves through different processing stages. Each stage
transforms the data into a new state, ensuring type safety and preventing
invalid state transitions.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import MappingProxyType
import typing

# --- Minimal guard helpers (clarity > boilerplate) ---

T = typing.TypeVar("T")


def _freeze_mapping(
    m: dict[str, T] | typing.Mapping[str, T] | None,
) -> typing.Mapping[str, T] | None:
    """Return an immutable mapping view or None.

    Accepts dict or Mapping; wraps dicts in MappingProxyType while preserving type.
    """
    if m is None or isinstance(m, MappingProxyType):
        return m
    return MappingProxyType(dict(m))


def _is_tuple_of(value: object, typ: type | tuple[type, ...]) -> bool:
    return isinstance(value, tuple) and all(isinstance(v, typ) for v in value)


def _require(
    *,
    condition: bool,
    message: str,
    exc: type[Exception] = ValueError,
    field_name: str | None = None,
) -> None:
    """Centralized validation with optional field context for clearer errors."""
    if not condition:
        if field_name:
            enhanced_message = f"{field_name}: {message}"
            raise exc(enhanced_message)
        raise exc(message)


def _require_zero_arg_callable(func: typing.Any, field_name: str) -> None:
    """Validate callable takes no arguments for predictable execution."""
    _require(
        condition=callable(func),
        message="must be callable",
        field_name=field_name,
        exc=TypeError,
    )

    # Validate signature if introspectable
    try:
        sig = inspect.signature(func)
        has_required_params = any(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            for p in sig.parameters.values()
        )
        _require(
            condition=not has_required_params,
            message="must be a zero-argument callable",
            field_name=field_name,
            exc=TypeError,
        )
    except (ValueError, RuntimeError):
        # Some callables may not have introspectable signatures; acceptable
        # Note: Only catch signature inspection errors, not validation failures
        pass


# Note: This import is for type-checking/static analysis only and ensures
# external dependency accounting tests detect the SDK dependency without
# introducing a runtime import in core modules.
if typing.TYPE_CHECKING:  # pragma: no cover - import for dependency visibility only
    from collections.abc import Callable

    # Ensure external dependency visibility for tests without runtime import
    from google.genai import types as genai_types  # noqa: F401

    from gemini_batch.config import FrozenConfig

# --- Result Monad for Robust Error Handling ---
# Implements the "Unified Result Type" enhancement for explicit error handling.
# This prevents the need for broad try/except blocks in the executor and makes
# failures a predictable part of the data flow.

TSuccess = typing.TypeVar("TSuccess")
TFailure = typing.TypeVar("TFailure", bound=Exception)


@dataclasses.dataclass(frozen=True, slots=True)
class Success[TSuccess]:
    """A successful result in the pipeline."""

    value: TSuccess


@dataclasses.dataclass(frozen=True, slots=True)
class Failure[TFailure]:
    """A failure in the pipeline, containing the error."""

    error: TFailure


Result = Success[TSuccess] | Failure[TFailure]

# --- Core Data Models ---


@dataclasses.dataclass(frozen=True, slots=True)
class ConversationTurn:
    """A single turn in a conversation history."""

    question: str
    answer: str
    is_error: bool = False

    def __post_init__(self) -> None:
        """Validate invariants for type safety."""
        _require(
            condition=isinstance(self.question, str),
            message="must be str",
            field_name="question",
            exc=TypeError,
        )
        _require(
            condition=isinstance(self.answer, str),
            message="must be str",
            field_name="answer",
            exc=TypeError,
        )
        # Prevent degenerate conversation turns
        _require(
            condition=not (self.question.strip() == "" and self.answer.strip() == ""),
            message="cannot both be empty (after stripping whitespace)",
            field_name="question and answer",
        )


@dataclasses.dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:
        """Validate Source invariants and loader signature."""
        _require(
            condition=self.source_type in ("text", "youtube", "arxiv", "file"),
            message=f"must be one of ['text','youtube','arxiv','file'], got {self.source_type!r}",
            field_name="source_type",
        )
        _require(
            condition=isinstance(self.identifier, str | Path),
            message="must be str | Path",
            field_name="identifier",
            exc=TypeError,
        )
        # Additional validation for Path identifiers
        if isinstance(self.identifier, Path):
            _require(
                condition=str(self.identifier).strip() != "",
                message="Path cannot be empty",
                field_name="identifier",
            )
        elif isinstance(self.identifier, str):
            _require(
                condition=self.identifier.strip() != "",
                message="cannot be empty string",
                field_name="identifier",
            )

        _require(
            condition=isinstance(self.mime_type, str) and self.mime_type.strip() != "",
            message="must be a non-empty str",
            field_name="mime_type",
            exc=TypeError,
        )
        _require(
            condition=isinstance(self.size_bytes, int) and self.size_bytes >= 0,
            message="must be an int >= 0",
            field_name="size_bytes",
        )

        # Use the dedicated helper for callable validation
        _require_zero_arg_callable(self.content_loader, "content_loader")

    # --- Ergonomic constructors for common cases ---
    @classmethod
    def from_text(cls, content: str, identifier: str | None = None) -> Source:
        """Create a text `Source` from a string.

        Args:
            content: Text content to analyze.
            identifier: Optional identifier; defaults to a snippet of the content.

        Returns:
            A `Source` representing UTF-8 encoded text.
        """
        _require(
            condition=isinstance(content, str),
            message="must be a str",
            field_name="content",
            exc=TypeError,
        )
        encoded = content.encode("utf-8")
        display = identifier if identifier is not None else content[:100]
        return cls(
            source_type="text",
            identifier=display,
            mime_type="text/plain",
            size_bytes=len(encoded),
            content_loader=lambda: encoded,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Source:
        """Create a file `Source` from a local filesystem path.

        Args:
            path: Path to a local file.

        Returns:
            A `Source` that lazily loads the file bytes.
        """
        file_path = Path(path)
        _require(
            condition=file_path.is_file(),
            message="path must point to an existing file",
            field_name="path",
        )
        import mimetypes

        mime_type, _ = mimetypes.guess_type(str(file_path))
        return cls(
            source_type="file",
            identifier=file_path,
            mime_type=mime_type or "application/octet-stream",
            size_bytes=file_path.stat().st_size,
            content_loader=lambda: file_path.read_bytes(),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class PromptBundle:
    """Immutable container for assembled prompts with provenance.

    This represents the final, composed prompts after assembly from various
    sources (inline config, files, builder hooks). The bundle preserves the
    exact count of user prompts to maintain batching invariants while allowing
    system instructions to be added separately.
    """

    user: tuple[
        str, ...
    ]  # Transformed user prompts (prefix/suffix applied), count preserved
    system: str | None = None  # Optional system instruction
    hints: typing.Mapping[str, typing.Any] = dataclasses.field(
        default_factory=dict
    )  # Provenance flags (has_sources, user_from, etc.)

    def __post_init__(self) -> None:
        """Validate and freeze prompt bundle components."""
        # Ensure user prompts are strings and not empty
        _require(
            condition=_is_tuple_of(self.user, str),
            message="must be a tuple[str, ...]",
            field_name="user",
            exc=TypeError,
        )
        _require(
            condition=len(self.user) > 0,
            message="must contain at least one prompt",
            field_name="user",
        )
        # Validate that user prompts are not all empty
        non_empty_prompts = [p for p in self.user if p.strip()]
        _require(
            condition=len(non_empty_prompts) > 0,
            message="must contain at least one non-empty prompt",
            field_name="user",
        )

        _require(
            condition=self.system is None or isinstance(self.system, str),
            message="must be a str or None",
            field_name="system",
            exc=TypeError,
        )
        # System prompt should not be empty string if provided
        if self.system is not None:
            _require(
                condition=self.system.strip() != "",
                message="cannot be empty string when provided",
                field_name="system",
            )

        # Freeze hints mapping to prevent downstream mutation
        frozen = _freeze_mapping(self.hints)
        if frozen is not None:
            object.__setattr__(self, "hints", frozen)


# --- Typed Command States ---
# These dataclasses define the shape of our data as it is transformed by
# each stage of the pipeline.


@dataclasses.dataclass(frozen=True, slots=True)
class TokenEstimate:
    """Range-based token estimate with confidence.

    Models uncertainty explicitly through ranges, allowing conservative
    or optimistic decisions based on use case needs.
    """

    min_tokens: int
    expected_tokens: int
    max_tokens: int
    confidence: float
    breakdown: typing.Mapping[str, TokenEstimate] | None = None

    def __post_init__(self) -> None:
        """Validate invariants for ordering and bounds."""
        _require(
            condition=isinstance(self.min_tokens, int) and self.min_tokens >= 0,
            message=f"must be an int >= 0, got {self.min_tokens}",
            field_name="min_tokens",
        )
        _require(
            condition=isinstance(self.expected_tokens, int)
            and self.expected_tokens >= 0,
            message=f"must be an int >= 0, got {self.expected_tokens}",
            field_name="expected_tokens",
        )
        _require(
            condition=isinstance(self.max_tokens, int) and self.max_tokens >= 0,
            message=f"must be an int >= 0, got {self.max_tokens}",
            field_name="max_tokens",
        )
        _require(
            condition=self.min_tokens <= self.expected_tokens <= self.max_tokens,
            message=f"require min <= expected <= max, got {self.min_tokens} <= {self.expected_tokens} <= {self.max_tokens}",
            field_name="token ordering",
        )
        _require(
            condition=isinstance(self.confidence, int | float)
            and 0.0 <= self.confidence <= 1.0,
            message=f"must be numeric within [0.0, 1.0], got {self.confidence}",
            field_name="confidence",
        )
        # Freeze nested breakdown map if provided
        frozen = _freeze_mapping(self.breakdown)
        if frozen is not None:
            object.__setattr__(self, "breakdown", frozen)


@dataclasses.dataclass(frozen=True, slots=True)
class InitialCommand:
    """The initial state of a request, created by the user."""

    sources: tuple[typing.Any, ...]
    prompts: tuple[str, ...]
    config: FrozenConfig
    history: tuple[ConversationTurn, ...] = dataclasses.field(default_factory=tuple)
    # Extensions may attach immutable capsules; core remains agnostic to their types.
    # Pipeline stages interpret hints in a fail-soft manner (unknown hints ignored).
    hints: tuple[object, ...] | None = None

    def __post_init__(self) -> None:
        """Validate InitialCommand invariants."""
        _require(
            condition=isinstance(self.sources, tuple),
            message="must be a tuple",
            field_name="sources",
            exc=TypeError,
        )
        _require(
            condition=_is_tuple_of(self.prompts, str),
            message="must be a tuple[str, ...]",
            field_name="prompts",
            exc=TypeError,
        )
        # Ensure prompts structure is valid - content validation handled by prompt assembler
        _require(
            condition=self.prompts is not None,
            message="prompts field cannot be None",
            field_name="prompts",
        )
        # Prompts validation is handled by the prompt assembler, which has access to
        # configuration and can provide more specific error messages

        _require(
            condition=_is_tuple_of(self.history, ConversationTurn),
            message="must be a tuple[ConversationTurn, ...]",
            field_name="history",
            exc=TypeError,
        )
        _require(
            condition=self.hints is None or isinstance(self.hints, tuple),
            message="must be a tuple[object, ...] or None",
            field_name="hints",
            exc=TypeError,
        )

    # Strict construction helper for friendlier early failures
    @classmethod
    def strict(
        cls,
        *,
        sources: tuple[typing.Any, ...],
        prompts: tuple[str, ...],
        config: FrozenConfig,
        history: tuple[ConversationTurn, ...] = (),
        hints: tuple[object, ...] | None = None,
    ) -> InitialCommand:
        """Construct an `InitialCommand` ensuring at least one non-empty prompt.

        This surfaces prompt validity issues at creation time rather than during
        prompt assembly, improving onboarding and error locality.
        """
        _require(
            condition=isinstance(prompts, tuple)
            and all(isinstance(p, str) for p in prompts),
            message="prompts must be a tuple[str, ...]",
            field_name="prompts",
            exc=TypeError,
        )
        _require(
            condition=any((p or "").strip() for p in prompts),
            message="must contain at least one non-empty prompt",
            field_name="prompts",
        )
        return cls(
            sources=sources,
            prompts=prompts,
            config=config,
            history=history,
            hints=hints,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ResolvedCommand:
    """The state after sources have been resolved."""

    initial: InitialCommand
    resolved_sources: tuple[Source, ...]


# --- Library-owned neutral API payload types ---


@dataclasses.dataclass(frozen=True, slots=True)
class TextPart:
    """A minimal library-owned representation of a text part.

    Additional part types (e.g., images, files) can be added later while
    keeping the core decoupled from any vendor SDK types.
    """

    text: str

    def __post_init__(self) -> None:
        """Validate TextPart invariants."""
        _require(
            condition=isinstance(self.text, str),
            message="text must be a str",
            exc=TypeError,
        )


# Neutral union of API parts; extendable without leaking provider types
type APIPart = (
    TextPart | "FileRefPart" | "FilePlaceholder" | "HistoryPart" | "FileInlinePart"
)


@dataclasses.dataclass(frozen=True, slots=True)
class FileRefPart:
    """Provider-agnostic reference to uploaded content."""

    uri: str
    mime_type: str | None = None
    # Preserve raw provider object for advanced use cases
    raw_provider_data: typing.Any = None

    def __post_init__(self) -> None:
        """Validate FileRefPart invariants."""
        _require(
            condition=isinstance(self.uri, str) and self.uri != "",
            message="uri must be a non-empty str",
            exc=TypeError,
        )
        _require(
            condition=self.mime_type is None or isinstance(self.mime_type, str),
            message="mime_type must be a str or None",
            exc=TypeError,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class FileInlinePart:
    """Neutral inline file content for provider-agnostic cache creation.

    Carries raw bytes plus MIME type. Intended for opportunistic creation of
    cached contents where reading the file once at execution time is acceptable.
    """

    mime_type: str
    data: bytes

    def __post_init__(self) -> None:
        """Validate FileInlinePart invariants."""
        _require(
            condition=isinstance(self.mime_type, str) and self.mime_type.strip() != "",
            message="mime_type must be a non-empty str",
            exc=TypeError,
        )
        _require(
            condition=isinstance(self.data, bytes | bytearray),
            message="data must be bytes-like",
            exc=TypeError,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class FilePlaceholder:
    """Placeholder for a local file intended to be uploaded by the provider.

    Handlers replace this with a `FileRefPart` when uploads are supported.
    """

    local_path: Path
    mime_type: str | None = None

    def __post_init__(self) -> None:
        """Validate FilePlaceholder invariants."""
        _require(
            condition=isinstance(self.local_path, Path),
            message="local_path must be a pathlib.Path",
            exc=TypeError,
        )
        _require(
            condition=self.mime_type is None or isinstance(self.mime_type, str),
            message="mime_type must be a str or None",
            exc=TypeError,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class HistoryPart:
    """Structured conversation history to prepend to prompts.

    The pipeline preserves this structured part intact. Provider adapters are
    responsible for rendering it into provider-specific types (e.g., into a
    textual representation when needed). This keeps the planner/handler
    data-centric and decoupled from provider SDK details.
    """

    turns: tuple[ConversationTurn, ...]

    def __post_init__(self) -> None:
        """Validate HistoryPart shape."""
        _require(
            condition=_is_tuple_of(self.turns, ConversationTurn),
            message="turns must be a tuple[ConversationTurn, ...]",
            exc=TypeError,
        )

    @classmethod
    def from_raw_history(cls, raw_turns: typing.Any) -> typing.Self:
        """Validate and normalize raw history data into a HistoryPart.

        Accepted item shapes per turn (strict):
        - ConversationTurn instance (used as-is)
        - Mapping with string keys 'question' and 'answer' whose values are str
        - Object with string attributes 'question' and 'answer'

        Empty or None input yields an empty history. Invalid items raise
        ValueError/TypeError with precise index and reason. Both fields cannot
        be empty strings simultaneously.
        """
        if not raw_turns:
            return cls(turns=())

        validated: list[ConversationTurn] = []
        for i, raw in enumerate(raw_turns):
            try:
                if isinstance(raw, ConversationTurn):
                    q_val: object | None = raw.question
                    a_val: object | None = raw.answer
                elif isinstance(raw, dict):
                    # Prefer concise structural handling; mapping pattern keeps it clear.
                    q_val = raw.get("question")
                    a_val = raw.get("answer")
                else:
                    q_val = getattr(raw, "question", None)
                    a_val = getattr(raw, "answer", None)

                if not isinstance(q_val, str) or not isinstance(a_val, str):
                    raise TypeError("'question' and 'answer' must be str")
                if q_val == "" and a_val == "":
                    raise ValueError("question and answer cannot both be empty")

                # After the checks above, both are str for type checkers
                validated.append(ConversationTurn(question=q_val, answer=a_val))
            except Exception as e:
                raise ValueError(f"Invalid conversation turn at index {i}: {e}") from e

        return cls(turns=tuple(validated))


@dataclasses.dataclass(frozen=True, slots=True)
class RateConstraint:
    """Immutable rate limit specification.

    All rates are per-minute.

    Attributes:
        requests_per_minute (int): Number of requests allowed per minute (>0).
        tokens_per_minute (int | None): Optional tokens-per-minute (>0 if provided).
        min_interval_ms (int): Minimum interval between requests in milliseconds (>=0).
        burst_factor (float): Multiplier for burst capacity (>=1.0).
    """

    requests_per_minute: int
    tokens_per_minute: int | None = None
    min_interval_ms: int = 0
    burst_factor: float = 1.0

    def __post_init__(self) -> None:
        """Validate provided values; reject invalid inputs explicitly.

        Use the centralized `_require` helpers for consistent, contextual
        validation errors (type vs value concerns separated where helpful).
        """
        # requests_per_minute: must be int and > 0
        _require(
            condition=isinstance(self.requests_per_minute, int),
            message="must be an int",
            field_name="requests_per_minute",
            exc=TypeError,
        )
        _require(
            condition=self.requests_per_minute > 0,
            message="must be > 0",
            field_name="requests_per_minute",
        )

        # tokens_per_minute: optional int > 0 when provided
        _require(
            condition=self.tokens_per_minute is None
            or isinstance(self.tokens_per_minute, int),
            message="must be an int or None",
            field_name="tokens_per_minute",
            exc=TypeError,
        )
        if self.tokens_per_minute is not None:
            _require(
                condition=self.tokens_per_minute > 0,
                message="must be > 0 when provided",
                field_name="tokens_per_minute",
            )

        # min_interval_ms: int >= 0  # noqa: ERA001
        _require(
            condition=isinstance(self.min_interval_ms, int),
            message="must be an int",
            field_name="min_interval_ms",
            exc=TypeError,
        )
        _require(
            condition=self.min_interval_ms >= 0,
            message="must be >= 0",
            field_name="min_interval_ms",
        )

        # burst_factor: numeric (int|float) and >= 1.0
        _require(
            condition=isinstance(self.burst_factor, int | float),
            message="must be numeric (int|float)",
            field_name="burst_factor",
            exc=TypeError,
        )
        _require(
            condition=self.burst_factor >= 1.0,
            message="must be >= 1.0",
            field_name="burst_factor",
        )


# For the minimal slice, a generation config is simply a mapping. We keep it
# library-owned and translate to provider-specific types at the API seam.
GenerationConfigDict = typing.Mapping[str, object]


@dataclasses.dataclass(frozen=True, slots=True)
class APICall:
    """A description of a single API call to be made."""

    model_name: str
    api_parts: tuple[APIPart, ...]
    api_config: GenerationConfigDict
    # Optional best-effort cache name; adapters that support caching may reuse
    # this name. Callers must remain correct when caching is unsupported.
    cache_name_to_use: str | None = None

    def __post_init__(self) -> None:
        """Validate APICall invariants and freeze config mapping."""
        _require(
            condition=isinstance(self.model_name, str) and self.model_name != "",
            message="model_name must be a non-empty str",
            exc=TypeError,
        )
        _require(
            condition=isinstance(self.api_parts, tuple),
            message="api_parts must be a tuple",
            exc=TypeError,
        )
        valid_types = (
            TextPart,
            FileRefPart,
            FilePlaceholder,
            HistoryPart,
            FileInlinePart,
        )
        for idx, part in enumerate(self.api_parts):
            _require(
                condition=isinstance(part, valid_types),
                message=f"api_parts[{idx}] has invalid type {type(part)}; expected one of {valid_types!r}",
                exc=TypeError,
            )
        frozen = _freeze_mapping(self.api_config)
        if frozen is not None:
            object.__setattr__(self, "api_config", frozen)
        _require(
            condition=self.cache_name_to_use is None
            or isinstance(self.cache_name_to_use, str),
            message="cache_name_to_use must be a str or None",
            exc=TypeError,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Instructions for executing API calls.

    This includes both the primary call and an optional fallback call
    for when the primary call fails.
    """

    primary_call: APICall
    fallback_call: APICall | None = None  # For when batching fails
    # Vectorized execution: N independent calls with shared context
    calls: tuple[APICall, ...] = ()
    shared_parts: tuple[APIPart, ...] = ()
    # Optional rate limiting constraint
    rate_constraint: RateConstraint | None = None
    # Optional pre-generation actions
    upload_tasks: tuple[UploadTask, ...] = ()

    def __post_init__(self) -> None:
        """Validate plan collections and optionals."""
        # Basic integrity checks on collections
        _require(
            condition=_is_tuple_of(self.calls, APICall),
            message="calls must be a tuple[APICall, ...]",
            exc=TypeError,
        )
        _require(
            condition=_is_tuple_of(
                self.shared_parts,
                (TextPart, FileRefPart, FilePlaceholder, HistoryPart, FileInlinePart),
            ),
            message="shared_parts must be a tuple[APIPart, ...]",
            exc=TypeError,
        )
        _require(
            condition=self.rate_constraint is None
            or isinstance(self.rate_constraint, RateConstraint),
            message="rate_constraint must be RateConstraint or None",
            exc=TypeError,
        )
        _require(
            condition=_is_tuple_of(self.upload_tasks, UploadTask),
            message="upload_tasks must be a tuple[UploadTask, ...]",
            exc=TypeError,
        )
        # No cache planning fields are present in ExecutionPlan.


@dataclasses.dataclass(frozen=True, slots=True)
class UploadTask:
    """Instruction to upload a local file and substitute an API part."""

    part_index: int
    local_path: Path
    mime_type: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        """Validate UploadTask invariants."""
        _require(
            condition=isinstance(self.part_index, int) and self.part_index >= 0,
            message="part_index must be an int >= 0",
        )
        _require(
            condition=isinstance(self.local_path, Path),
            message="local_path must be a pathlib.Path",
            exc=TypeError,
        )
        _require(
            condition=self.mime_type is None or isinstance(self.mime_type, str),
            message="mime_type must be a str or None",
            exc=TypeError,
        )
        _require(
            condition=isinstance(self.required, bool),
            message="required must be a bool",
            exc=TypeError,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class PlannedCommand:
    """The state after an execution plan has been created."""

    resolved: ResolvedCommand
    execution_plan: ExecutionPlan
    token_estimate: TokenEstimate | None = None
    # Per-call estimates for vectorized plans
    per_call_estimates: tuple[TokenEstimate, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
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
