"""Extensions for Gemini batch processing framework.

High-level, data-centric add-ons that build on the core pipeline while keeping
the single execution seam intact. Extensions favor immutable data, pure
compilation, and minimal façades for excellent auditability and DX.

Extensions overview:
- Conversation: Available. Multi-turn conversation with planning, modes, and store-backed engine.
- Visualization: Upcoming. Notebook-first helpers under `gemini_batch.extensions.visualization`.
- Token Counting: Upcoming. Provider-aware token counting under `gemini_batch.extensions.token_counting`.

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
]

# Note: Visualization and Token Counting are intentionally not re-exported here yet.
# They are tracked as upcoming extensions to keep this PR focused on Conversation.
