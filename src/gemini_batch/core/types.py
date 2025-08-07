"""Core data types that flow through the pipeline.

This module defines the immutable data structures that represent the state
of a request as it moves through different processing stages. Each stage
transforms the data into a new state, ensuring type safety and preventing
invalid state transitions.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

from google.genai import types as genai_types

from gemini_batch.config import GeminiConfig

# --- Result Monad for Robust Error Handling ---
# Implements the "Unified Result Type" enhancement for explicit error handling.
# This prevents the need for broad try/except blocks in the executor and makes
# failures a predictable part of the data flow.

TSuccess = TypeVar("TSuccess")
TFailure = TypeVar("TFailure", bound=Exception)


@dataclass(frozen=True)
class Success[TSuccess]:
    """A successful result in the pipeline."""

    value: TSuccess


@dataclass(frozen=True)
class Failure[TFailure]:
    """A failure in the pipeline, containing the error."""

    error: TFailure


Result = Success[TSuccess] | Failure[TFailure]

# --- Core Data Models ---


@dataclass(frozen=True)
class ConversationTurn:
    """A single turn in a conversation history."""

    question: str
    answer: str
    is_error: bool = False


@dataclass(frozen=True)
class Source:
    """A structured representation of a single input source.

    Content access is lazy via the `content_loader` callable to optimize
    memory usage, ensuring content is only loaded when needed.
    """

    source_type: Literal["text", "youtube", "arxiv", "file"]
    identifier: str | Path  # The original path, URL, or text identifier
    mime_type: str
    size_bytes: int
    content_loader: Callable[[], bytes]  # A function to get content on demand


# --- Typed Command States ---
# These dataclasses define the shape of our data as it is transformed by
# each stage of the pipeline.


@dataclass(frozen=True)
class InitialCommand:
    """The initial state of a request, created by the user."""

    sources: tuple[Any, ...]
    prompts: tuple[str, ...]
    config: GeminiConfig
    history: tuple[ConversationTurn, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ResolvedCommand:
    """The state after sources have been resolved."""

    initial: InitialCommand
    resolved_sources: tuple[Source, ...]


@dataclass(frozen=True)
class APICall:
    """A description of a single API call to be made."""

    model_name: str
    api_parts: list[genai_types.Part]
    api_config: genai_types.GenerationConfig
    cache_name_to_use: str | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    """Instructions for executing API calls.

    This includes both the primary call and an optional fallback call
    for when the primary call fails.
    """

    primary_call: APICall
    fallback_call: APICall | None = None  # For when batching fails


@dataclass(frozen=True)
class PlannedCommand:
    """The state after an execution plan has been created."""

    resolved: ResolvedCommand
    execution_plan: ExecutionPlan


@dataclass(frozen=True)
class FinalizedCommand:
    """The state after API calls have been executed."""

    planned: PlannedCommand
    raw_api_response: Any
    # This will be populated by a future Telemetry handler/context
    telemetry_data: dict = field(default_factory=dict)  # type: ignore
