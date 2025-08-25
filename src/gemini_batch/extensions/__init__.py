"""Data-centric conversation extensions for Gemini batch processing.

This module provides a minimal, data-driven conversation extension that implements
the A+B hybrid design with single pipeline seam. The extension focuses on multi-turn
conversations with advanced batch processing while delegating complexity to the core.

Key components:
- Conversation: Tiny facade for conversation operations
- ConversationPolicy: Immutable policy controlling behavior
- PromptSet: Prompts with execution mode (single/sequential/vectorized)
- ConversationPlan: Compiled execution plan for auditability
- compile_conversation: Pure function for plan compilation
- execute_plan: Single pipeline seam for execution

Architecture principles:
- Data-centric design with immutable state
- Pure compile-then-execute pattern
- Single pipeline seam via GeminiExecutor
- Minimal facade with essential operations
- Full alignment with architecture rubric

Example:
    from gemini_batch import create_executor
    from gemini_batch.extensions import Conversation, PromptSet

    executor = create_executor()
    conv = Conversation.start(executor)

    # Simple usage
    conv = await conv.ask("Hello")

    # Advanced batch with policy
    from gemini_batch.extensions import ConversationPolicy
    policy = ConversationPolicy.cost_saver()
    conv, answers, metrics = await conv.with_policy(policy).run(
        PromptSet.vec("Q1", "Q2", "Q3")
    )
"""

from .conversation import (
    Conversation,
    ConversationState,
    Exchange,
)
from .conversation_planner import ConversationPlan, compile_conversation
from .conversation_types import (
    BatchMetrics,
    ConversationAnalytics,
    ConversationPolicy,
    PromptSet,
)

__all__ = [  # noqa: RUF022
    # Core conversation facade
    "Conversation",
    "ConversationState",
    "Exchange",
    # Data-centric types
    "BatchMetrics",
    "ConversationAnalytics",
    "ConversationPolicy",
    "PromptSet",
    "ConversationPlan",
    # Planning
    "compile_conversation",
]
