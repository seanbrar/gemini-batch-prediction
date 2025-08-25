# src/gemini_batch/extensions/conversation_engine.py
from __future__ import annotations

from typing import TYPE_CHECKING

from .conversation import Conversation

if TYPE_CHECKING:
    from gemini_batch.executor import GeminiExecutor

    from .conversation_store import ConversationStore
    from .conversation_types import ConversationState, Exchange


class ConversationEngine:
    def __init__(self, executor: GeminiExecutor, store: ConversationStore) -> None:
        self._executor = executor
        self._store = store

    async def ask(self, conversation_id: str, prompt: str) -> Exchange:
        state: ConversationState = await self._store.load(conversation_id)
        conv = Conversation(state, self._executor)
        conv2 = await conv.ask(prompt)
        # OCC append (store enforces expected_version)
        await self._store.append(
            conversation_id, expected_version=state.version, ex=conv2.state.turns[-1]
        )
        return conv2.state.turns[-1]
