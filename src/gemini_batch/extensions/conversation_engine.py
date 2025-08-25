from __future__ import annotations

from typing import TYPE_CHECKING

from .conversation import ConversationState, Exchange, extend

if TYPE_CHECKING:
    from gemini_batch.executor import GeminiExecutor

    from .conversation_store import ConversationStore


class ConversationEngine:
    """Backend-friendly engine with optimistic concurrency via a store."""

    def __init__(self, executor: GeminiExecutor, store: ConversationStore) -> None:
        self._executor = executor
        self._store = store

    async def ask(self, conversation_id: str, prompt: str) -> Exchange:
        state: ConversationState = await self._store.load(conversation_id)
        new_state, ex, _ = await extend(state, prompt, self._executor)
        # OCC append; if conflict, let it surface to caller
        await self._store.append(conversation_id, expected_version=state.version, ex=ex)
        return ex
