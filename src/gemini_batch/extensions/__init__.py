"""Extensions for Gemini batch processing framework.

High-level, data-centric add-ons that build on the core pipeline while keeping
the single execution seam intact. Extensions favor immutable data, pure
compilation, and minimal façades for excellent auditability and DX.

Extensions overview:
- Conversation: Available. Multi-turn conversation with planning, modes, and store-backed engine.
- Visualization: Available. Notebook-first helpers under `gemini_batch.extensions.visualization`.
- Token Counting: Available. Provider-aware token counting under `gemini_batch.extensions.token_counting`.
- Chunking: Available. Text and transcript chunkers under `gemini_batch.extensions.chunking`.
- Timestamp Linker: Available. Answer timestamp extraction under `gemini_batch.extensions.timestamp_linker`.
- Dependency Analyzer: Available. Preflight dependency hints under `gemini_batch.extensions.dependency_analyzer`.
- Model Selector: Available. Advisory model selection under `gemini_batch.extensions.model_selector`.

Architecture principles:
- Data-centric design with immutable state
- Pure compile-then-execute pattern
- Single pipeline seam via GeminiExecutor
- Minimal façades with essential operations
- Aligned with Architecture Rubric (simplicity, clarity, robustness, DX, extensibility)

Quick start:
    from gemini_batch import create_executor
    from gemini_batch.extensions import Conversation, PromptSet

    executor = create_executor()
    conv = Conversation.start(executor)
    conv = await conv.ask("Hello")

    # Advanced batch with policy
    from gemini_batch.extensions import ConversationPolicy
    policy = ConversationPolicy.cost_saver()
    conv, answers, metrics = await conv.with_policy(policy).run(
        PromptSet.vectorized("Q1", "Q2", "Q3")
    )
"""

# Expose plugin submodules for convenient imports
from . import chunking as chunking
from . import dependency_analyzer as dependency_analyzer
from . import model_selector as model_selector
from . import timestamp_linker as timestamp_linker
from . import token_counting as token_counting
from . import visualization as visualization
from .conversation import Conversation
from .conversation_engine import ConversationEngine
from .conversation_planner import ConversationPlan, compile_conversation
from .conversation_store import ConversationStore, JSONStore
from .conversation_types import (
    BatchMetrics,
    ConversationAnalytics,
    ConversationPolicy,
    ConversationState,
    Exchange,
    PromptSet,
)

__all__ = [  # noqa: RUF022
    # Conversation extension
    "Conversation",
    "ConversationState",
    "Exchange",
    "BatchMetrics",
    "ConversationAnalytics",
    "ConversationPolicy",
    "PromptSet",
    "ConversationPlan",
    "compile_conversation",
    # Advanced ergonomics
    "ConversationEngine",
    "ConversationStore",
    "JSONStore",
    # Plugin modules (importable via `from gemini_batch.extensions import X`)
    "chunking",
    "dependency_analyzer",
    "model_selector",
    "timestamp_linker",
    "token_counting",
    "visualization",
    # Token counting extension (exported for convenience)
    "GeminiTokenCounter",
    "ValidContent",
    "TokenCountSuccess",
    "TokenCountFailure",
    "TokenCountResult",
    "TokenCountError",
    "InvalidContentError",
    "ErrorInfo",
    "count_gemini_tokens",
    # Convenience re-exports for other plugins
    # chunking
    "TokenEstimator",
    "TranscriptSegment",
    "TranscriptChunk",
    "chunk_text_by_tokens",
    "chunk_transcript_by_tokens",
    # dependency analyzer
    "analyze_dependencies",
    # model selector
    "SelectionInputs",
    "SelectionDecision",
    "decide",
    "maybe_override_model",
    # timestamp linker
    "link_timestamps",
    # visualization facade (lightweight pure helpers are safe to export)
    "summarize_efficiency",
    "create_efficiency_visualizations",
    "create_focused_efficiency_visualization",
    "visualize_scaling_results",
]

# Token Counting and other plugins are exported for convenience as documented.

# Re-exports to keep import surface tidy
# Convenience re-exports for other plugin APIs
from .chunking import (
    TokenEstimator,
    TranscriptChunk,
    TranscriptSegment,
    chunk_text_by_tokens,
    chunk_transcript_by_tokens,
)
from .dependency_analyzer import analyze_dependencies
from .model_selector import (
    SelectionDecision,
    SelectionInputs,
    decide,
    maybe_override_model,
)
from .timestamp_linker import link_timestamps
from .token_counting import (
    ErrorInfo,
    GeminiTokenCounter,
    InvalidContentError,
    TokenCountError,
    TokenCountFailure,
    TokenCountResult,
    TokenCountSuccess,
    ValidContent,
    count_gemini_tokens,
)
from .visualization import (
    create_efficiency_visualizations,
    create_focused_efficiency_visualization,
    summarize_efficiency,
    visualize_scaling_results,
)
