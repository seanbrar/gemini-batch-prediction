# Conversation Guide (Extension)

The Conversation extension provides a tiny, immutable API for multi‑turn interactions that composes the core pipeline. It maintains context across turns, supports sequential and vectorized execution, and surfaces audit‑friendly metrics — without modifying the core library.

## Quick Start

```python
import asyncio
from gemini_batch import create_executor
from gemini_batch.core.types import Source
from gemini_batch.extensions import Conversation, PromptSet

async def main():
    ex = create_executor()
    # Use explicit Sources (provider‑agnostic, validated)
    sources = [
        Source.from_file("document.pdf"),
        Source.from_text("Pinned context for this session", identifier="preamble"),
    ]
    conv = Conversation.start(ex, sources=sources)  # immutable facade

    # Single turn
    conv = await conv.ask("What is this about?")

    # Sequential turns in one call
    conv, answers, metrics = await conv.run(PromptSet.sequential("Q1", "Q2"))
    print(answers, metrics.totals)

    # Vectorized batch (one synthetic turn)
    conv, answers, _ = await conv.run(PromptSet.vectorized("A", "B", "C"))
    print(conv.state.turns[-1].user)  # => "[vectorized x3]"

asyncio.run(main())
```

## Core Concepts

- Immutable state: `ConversationState` is never mutated; every operation returns a new `Conversation` instance with an incremented `version`.
- Single pipeline seam: all execution flows through `executor.execute(InitialCommand)`; the extension never calls provider SDKs directly.
- Modes as data: `PromptSet` carries both prompts and the mode (`Single`, `Sequential`, `Vectorized`), which determines strategy and result formatting.
- Policy: `ConversationPolicy` influences planning hints (estimation/result/cache overrides) and history windowing.

## API Surface

- `Conversation.start(executor, sources=()) -> Conversation`
- `Conversation.with_sources(sources) -> Conversation`
- `Conversation.with_policy(policy) -> Conversation`
- `await Conversation.ask(prompt: str) -> Conversation`
- `await Conversation.run(prompt_set: PromptSet) -> (Conversation, tuple[str, ...], BatchMetrics)`
- `Conversation.analytics() -> ConversationAnalytics`

## Sources

```python
from gemini_batch.core.types import Source

# Replace sources immutably with explicit `Source` objects
conv = conv.with_sources([
    Source.from_file("paper1.pdf"),
    Source.from_file("paper2.pdf"),
])  # returns new instance

# Text context can be pinned via `Source.from_text`
conv = conv.with_sources([
    Source.from_text("You are a helpful assistant.", identifier="sys"),
])
```

## Modes

```python
from gemini_batch.extensions import PromptSet

# Single (one Q→A turn)
await conv.run(PromptSet.single("Hello"))

# Sequential (n Q→A turns)
await conv.run(PromptSet.sequential("Q1", "Q2"))

# Vectorized (one synthetic turn with combined answers)
await conv.run(PromptSet.vectorized("Q1", "Q2", "Q3"))
```

## Policy & Planning

```python
from gemini_batch.extensions import ConversationPolicy

policy = ConversationPolicy(
    keep_last_n=3,            # window history turns
    widen_max_factor=1.2,     # planner estimation widening
    clamp_max_tokens=16000,   # planner clamp
    prefer_json_array=True,   # extraction bias
    execution_cache_name="demo-cache",  # best-effort cache override
    reuse_cache_only=False,   # intent; provider decides capability
)

conv = conv.with_policy(policy)
```

`compile_conversation(state, prompt_set, policy)` produces a pure `ConversationPlan` with:

- `sources`, `history` (windowed), `prompts`
- `strategy`: `"sequential" | "vectorized"`
- `options`: structured `ExecutionOptions` (estimation/result/cache)
- `hints`: inspectable tuple mirroring options for audits/demos/tests

### Advanced: Cache identity & reuse

```python
# Attach a deterministic cache identity and artifacts to the state
state = conv.state
conv = Conversation(
    state.__class__(
        sources=state.sources,
        turns=state.turns,
        cache_key="demo:my‑cache‑key",
        cache_artifacts=("v1",),
        policy=state.policy,
        version=state.version,
    ),
    conv._Conversation__dict__["_executor"],  # internal; prefer start() in real code
)

# Prefer reuse of existing cache only (provider capability‑dependent)
conv = conv.with_policy(ConversationPolicy(reuse_cache_only=True))

# Best‑effort override to use a known provider cache name
conv = conv.with_policy(ConversationPolicy(execution_cache_name="cachedContents/xyz"))
```

## Metrics & Analytics

`run()` returns `BatchMetrics` with `per_prompt` and `totals`. The extension distributes totals if per‑prompt metrics aren’t provided. Token validation info is surfaced as warnings on the first exchange of a batch when significantly out of range.

```python
conv, answers, metrics = await conv.run(PromptSet.sequential("Q1", "Q2"))
print(metrics.per_prompt)
print(metrics.totals)
print(conv.analytics())  # lightweight conversation summary
```

## Persistence (optional)

For backends, use the store-backed engine to load → execute → append using optimistic concurrency.

```python
from gemini_batch.extensions.conversation_store import JSONStore
from gemini_batch.extensions.conversation_engine import ConversationEngine

store = JSONStore("./conversations.json")
engine = ConversationEngine(executor, store)

ex = await engine.ask("conv-123", "Hello?")
```

## Notes

- The extension is provider-neutral and never imports SDKs; all provider behavior flows through the core pipeline.
- Vectorized execution returns one synthetic exchange containing combined answers; sequential returns one exchange per prompt.

## Cheat Sheet: Strategies & Modes

- `PromptSet.single("...")` → strategy: `sequential`, one exchange
- `PromptSet.sequential("...", "...")` → strategy: `sequential`, N exchanges
- `PromptSet.vectorized("...", "...")` → strategy: `vectorized`, one synthetic exchange
