"""Immutable, data-first conversation extension with a tiny facade.

This module rebuilds the conversation extension to align with the
architecture rubric and ADR-0008. It provides:

- Immutable data types (`Exchange`, `ConversationState`)
- A pure `extend()` function that composes the pipeline via `GeminiExecutor`
- A simple `Conversation` facade with chainable `.ask()` and `.ask_many()`
- Backend components (`ConversationEngine`, `JSONStore`) for persistence with OCC
- Comprehensive analytics, branching, and context management features
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import TYPE_CHECKING, Any

from gemini_batch.core.types import ConversationTurn, InitialCommand
from gemini_batch.pipeline.hints import (
    CacheHint,
    EstimationOverrideHint,
    ExecutionCacheName,
    ResultHint,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterable

    from gemini_batch.executor import GeminiExecutor


logger = logging.getLogger(__name__)


# --- Immutable core types ---


@dataclass(frozen=True)
class Exchange:
    """A single question/answer pair with optional audit fields."""

    user: str
    assistant: str
    error: bool
    # Optional audit fields (populated when telemetry is available)
    estimate_min: int | None = None
    estimate_max: int | None = None
    actual_tokens: int | None = None
    in_range: bool | None = None


@dataclass(frozen=True)
class CacheBinding:
    """Conversation-scoped cache identity and artifacts (provider-agnostic)."""

    key: str
    artifacts: tuple[str, ...]
    ttl_seconds: int | None = None


@dataclass(frozen=True)
class ConversationHints:
    """Advanced hints for conversation-level control over pipeline behavior."""

    # Cost control and estimation
    widen_max_factor: float | None = None
    clamp_max_tokens: int | None = None

    # Result extraction preferences
    prefer_json_array: bool = False

    # Execution-time cache override
    execution_cache_name: str | None = None


@dataclass(frozen=True)
class ConversationState:
    """Immutable snapshot of a conversation."""

    sources: tuple[Any, ...]
    turns: tuple[Exchange, ...]
    cache: CacheBinding | None = None
    hints: ConversationHints | None = None
    version: int = 0

    @property
    def last(self) -> Exchange | None:
        return self.turns[-1] if self.turns else None


# --- Pure extension: state -> command -> result -> new state ---


async def extend(
    state: ConversationState,
    prompt: str,
    executor: GeminiExecutor,
    *,
    override_sources: tuple[Any, ...] | None = None,
    keep_last_n: int | None = None,
) -> tuple[ConversationState, Exchange, dict[str, Any]]:
    """Extend a conversation with a new user prompt and build a new snapshot.

    This function is pure with respect to the provided `state` and `executor`.
    It constructs an `InitialCommand` from the inputs, delegates execution to
    the single pipeline seam, and returns the next immutable state along with
    the `Exchange` produced for this turn.
    """
    cfg = executor.config
    frozen_config = cfg.to_frozen() if hasattr(cfg, "to_frozen") else cfg

    full_history: tuple[ConversationTurn, ...] = tuple(
        ConversationTurn(question=e.user, answer=e.assistant, is_error=e.error)
        for e in state.turns
    )
    history: tuple[ConversationTurn, ...]
    if keep_last_n is not None and keep_last_n >= 0:
        history = full_history[-keep_last_n:] if keep_last_n > 0 else ()
    else:
        history = full_history

    sources = override_sources if override_sources is not None else state.sources

    # Fail-soft: build hints from conversation state
    hints_list: list[object] = []

    # Cache hints from cache binding
    if state.cache is not None:
        hints_list.append(
            CacheHint(
                deterministic_key=state.cache.key,
                artifacts=state.cache.artifacts,
                ttl_seconds=state.cache.ttl_seconds,
                reuse_only=False,
            )
        )

    # Advanced hints from conversation hints
    if state.hints is not None:
        if (
            state.hints.widen_max_factor is not None
            or state.hints.clamp_max_tokens is not None
        ):
            hints_list.append(
                EstimationOverrideHint(
                    widen_max_factor=state.hints.widen_max_factor or 1.0,
                    clamp_max_tokens=state.hints.clamp_max_tokens,
                )
            )

        if state.hints.prefer_json_array:
            hints_list.append(ResultHint(prefer_json_array=True))

        if state.hints.execution_cache_name:
            hints_list.append(
                ExecutionCacheName(cache_name=state.hints.execution_cache_name)
            )

    hints: tuple[object, ...] = tuple(hints_list)

    command = InitialCommand(
        sources=sources,
        prompts=(prompt,),
        config=frozen_config,
        history=history,
        hints=hints,
    )

    result = await executor.execute(command)

    is_error = not bool(result.get("success", False))
    answers = result.get("answers")
    answer = (
        str(answers[0])
        if isinstance(answers, list) and answers
        else "Error: No answer found."
    )

    # Extract audit metrics, guarded for absence
    metrics = result.get("metrics") if isinstance(result, dict) else None
    token_validation = (
        metrics.get("token_validation") if isinstance(metrics, dict) else None
    )
    usage = result.get("usage") if isinstance(result, dict) else None

    estimate_min: int | None = None
    estimate_max: int | None = None
    actual_tokens: int | None = None
    in_range: bool | None = None

    if isinstance(token_validation, dict):
        estimate_min = _safe_int(token_validation.get("estimated_min"))
        estimate_max = _safe_int(token_validation.get("estimated_max"))
        in_range_val = token_validation.get("in_range")
        in_range = bool(in_range_val) if isinstance(in_range_val, bool) else None
        actual_tokens = _safe_int(token_validation.get("actual"))
    # Fallback for actual tokens when not present in token_validation
    if actual_tokens is None and isinstance(usage, dict):
        actual_tokens = _safe_int(
            usage.get("total_tokens") or usage.get("total_token_count")
        )

    ex = Exchange(
        user=prompt,
        assistant=answer,
        error=is_error,
        estimate_min=estimate_min,
        estimate_max=estimate_max,
        actual_tokens=actual_tokens,
        in_range=in_range,
    )

    new_state = ConversationState(
        sources=state.sources,
        turns=(*state.turns, ex),
        cache=state.cache,
        hints=state.hints,
        version=state.version + 1,
    )
    return new_state, ex, result


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


# --- Facade: tiny, immutable API ---


class Conversation:
    """Immutable, chainable facade over `ConversationState`."""

    def __init__(self, state: ConversationState, executor: GeminiExecutor):
        self._state = state
        self._executor = executor

    @classmethod
    def start(
        cls, executor: GeminiExecutor, *, sources: Iterable[Any] | None = None
    ) -> Conversation:
        return cls(
            ConversationState(sources=tuple(sources or ()), turns=(), hints=None),
            executor,
        )

    @property
    def state(self) -> ConversationState:
        return self._state

    # --- Cache binding (optional) ---
    def with_cache(self, *, key: str, ttl_seconds: int | None = None) -> Conversation:
        existing_artifacts: tuple[str, ...] = (
            self._state.cache.artifacts if self._state.cache else ()
        )
        new_state = replace(
            self._state,
            cache=CacheBinding(
                key=key, artifacts=existing_artifacts, ttl_seconds=ttl_seconds
            ),
            version=self._state.version + 1,
        )
        return Conversation(new_state, self._executor)

    def without_cache(self) -> Conversation:
        new_state = replace(self._state, cache=None, version=self._state.version + 1)
        return Conversation(new_state, self._executor)

    def with_hints(
        self,
        *,
        widen_max_factor: float | None = None,
        clamp_max_tokens: int | None = None,
        prefer_json_array: bool = False,
        execution_cache_name: str | None = None,
    ) -> Conversation:
        """Add advanced hints for pipeline behavior control.

        Args:
            widen_max_factor: Multiply max_tokens estimate by this factor for conservative planning
            clamp_max_tokens: Upper bound to clamp max_tokens after widening
            prefer_json_array: Bias toward JSON array extraction in result builder
            execution_cache_name: Override cache name at execution time
        """
        new_hints = ConversationHints(
            widen_max_factor=widen_max_factor,
            clamp_max_tokens=clamp_max_tokens,
            prefer_json_array=prefer_json_array,
            execution_cache_name=execution_cache_name,
        )
        new_state = replace(
            self._state,
            hints=new_hints,
            version=self._state.version + 1,
        )
        return Conversation(new_state, self._executor)

    def without_hints(self) -> Conversation:
        """Remove advanced hints from conversation."""
        new_state = replace(self._state, hints=None, version=self._state.version + 1)
        return Conversation(new_state, self._executor)

    # --- Source editing (persistent) ---
    def with_sources(self, sources: Iterable[Any]) -> Conversation:
        new_state = replace(
            self._state, sources=tuple(sources), version=self._state.version + 1
        )
        return Conversation(new_state, self._executor)

    def add_sources(self, *sources: Any) -> Conversation:
        current = list(self._state.sources)
        for s in sources:
            if s not in current:
                current.append(s)
        new_state = replace(
            self._state, sources=tuple(current), version=self._state.version + 1
        )
        return Conversation(new_state, self._executor)

    def remove_sources(self, predicate: Callable[[Any], bool]) -> Conversation:
        filtered = tuple(s for s in self._state.sources if not predicate(s))
        new_state = replace(
            self._state, sources=filtered, version=self._state.version + 1
        )
        return Conversation(new_state, self._executor)

    def replace_source(
        self, match: Callable[[Any], bool], new_source: Any
    ) -> Conversation:
        replaced: list[Any] = []
        replaced_once = False
        for s in self._state.sources:
            if not replaced_once and match(s):
                replaced.append(new_source)
                replaced_once = True
            else:
                replaced.append(s)
        new_state = replace(
            self._state, sources=tuple(replaced), version=self._state.version + 1
        )
        return Conversation(new_state, self._executor)

    # --- Asking (returns a NEW conversation) ---
    async def ask(
        self,
        prompt: str,
        *,
        sources: Iterable[Any] | None = None,
        keep_last_n: int | None = None,
    ) -> Conversation:
        effective_sources = (
            tuple(sources) if sources is not None else self._state.sources
        )
        new_state, _, _ = await extend(
            self._state,
            prompt,
            self._executor,
            override_sources=effective_sources,
            keep_last_n=keep_last_n,
        )
        return Conversation(new_state, self._executor)

    async def ask_many(self, *prompts: str) -> tuple[Conversation, tuple[str, ...]]:
        answers: list[str] = []
        conv: Conversation = self
        for p in prompts:
            conv = await conv.ask(p)
            last = conv.state.last
            answers.append(last.assistant if last else "")
        return conv, tuple(answers)

    async def ask_batch(
        self,
        prompts: Iterable[str],
        *,
        vectorized: bool = False,
        record_history: bool = True,
    ) -> tuple[Conversation, tuple[str, ...], BatchMetrics]:
        prompts_tuple = tuple(prompts)
        if not prompts_tuple:
            return self, (), BatchMetrics(per_prompt=(), totals={})

        if vectorized:
            # Single pipeline call, one history entry
            cfg = self._executor.config
            frozen_config = cfg.to_frozen() if hasattr(cfg, "to_frozen") else cfg
            history = tuple(
                ConversationTurn(question=e.user, answer=e.assistant, is_error=e.error)
                for e in self._state.turns
            )
            command = InitialCommand(
                sources=self._state.sources,
                prompts=prompts_tuple,
                config=frozen_config,
                history=history,
            )
            res = await self._executor.execute(command)
            answers = res.get("answers") if isinstance(res, dict) else None
            answers_tuple = (
                tuple(str(a) for a in answers) if isinstance(answers, list) else ()
            )

            # Optionally append a synthetic batch exchange (minimal, auditable)
            if record_history:
                assistant_text = "; ".join(answers_tuple) if answers_tuple else ""
                ex = Exchange(
                    user=f"[batch x{len(prompts_tuple)}]",
                    assistant=assistant_text,
                    error=not bool(res.get("success", False)),
                )
                new_state = ConversationState(
                    sources=self._state.sources,
                    turns=(*self._state.turns, ex),
                    cache=self._state.cache,
                    hints=self._state.hints,
                    version=self._state.version + 1,
                )
            else:
                new_state = self._state
            per_prompt: tuple[dict[str, int | float], ...] = tuple(
                {} for _ in answers_tuple
            )
            vectorized_totals = _metrics_from_result(res)
            return (
                Conversation(new_state, self._executor),
                answers_tuple,
                BatchMetrics(per_prompt=per_prompt, totals=vectorized_totals),
            )

        # Sequential fallback using extend()
        conv: Conversation = self
        answers_list: list[str] = []
        step_metrics: list[dict[str, int | float]] = []
        for p in prompts_tuple:
            new_state, ex, res = await extend(conv._state, p, conv._executor)
            conv = Conversation(new_state, conv._executor)
            answers_list.append(ex.assistant)
            step_metrics.append(_metrics_from_result(res))
        totals: dict[str, int | float] = {}
        for m in step_metrics:
            for k, v in m.items():
                totals[k] = totals.get(k, 0) + float(v)
        return (
            conv,
            tuple(answers_list),
            BatchMetrics(per_prompt=tuple(step_metrics), totals=totals),
        )

    async def run(
        self, flow: Flow
    ) -> tuple[Conversation, tuple[str, ...], FlowMetrics]:
        conv: Conversation = self
        answers: list[str] = []
        step_metrics: list[dict[str, int | float]] = []
        for p in flow.steps:
            new_state, ex, res = await extend(conv._state, p, conv._executor)
            conv = Conversation(new_state, conv._executor)
            answers.append(ex.assistant)
            step_metrics.append(_metrics_from_result(res))
        flow_totals: dict[str, int | float] = {}
        for m in step_metrics:
            for k, v in m.items():
                flow_totals[k] = flow_totals.get(k, 0) + float(v)
        return (
            conv,
            tuple(answers),
            FlowMetrics(steps=tuple(step_metrics), totals=flow_totals),
        )

    # --- Analytics and observability ---
    def analytics(self) -> ConversationAnalytics:
        """Compute comprehensive analytics for the conversation."""
        turns = self._state.turns
        total_turns = len(turns)

        if total_turns == 0:
            return ConversationAnalytics(
                total_turns=0,
                error_turns=0,
                success_rate=1.0,
            )

        error_turns = sum(1 for turn in turns if turn.error)
        success_rate = (total_turns - error_turns) / total_turns

        # Token metrics
        estimated_tokens = []
        actual_tokens = []
        for turn in turns:
            if turn.estimate_max is not None:
                estimated_tokens.append(turn.estimate_max)
            if turn.actual_tokens is not None:
                actual_tokens.append(turn.actual_tokens)

        total_estimated = sum(estimated_tokens) if estimated_tokens else None
        total_actual = sum(actual_tokens) if actual_tokens else None
        estimation_accuracy = (
            total_actual / total_estimated
            if total_estimated and total_actual and total_estimated > 0
            else None
        )

        # Content metrics
        total_user_chars = sum(len(turn.user) for turn in turns)
        total_assistant_chars = sum(len(turn.assistant) for turn in turns)
        avg_response_length = (
            total_assistant_chars / total_turns if total_turns > 0 else 0.0
        )

        return ConversationAnalytics(
            total_turns=total_turns,
            error_turns=error_turns,
            success_rate=success_rate,
            total_estimated_tokens=total_estimated,
            total_actual_tokens=total_actual,
            estimation_accuracy=estimation_accuracy,
            avg_response_length=avg_response_length,
            total_user_chars=total_user_chars,
            total_assistant_chars=total_assistant_chars,
        )

    def health_score(self) -> float:
        """Compute a health score (0.0-1.0) for the conversation."""
        analytics = self.analytics()

        # Base score on success rate
        score = analytics.success_rate

        # Penalize for poor estimation accuracy (if available)
        if analytics.estimation_accuracy is not None:
            # Ideal estimation accuracy is 1.0, deviations reduce score
            accuracy_penalty = abs(1.0 - analytics.estimation_accuracy) * 0.1
            score = max(0.0, score - accuracy_penalty)

        # Bonus for conversations with actual activity
        if analytics.total_turns > 0:
            activity_bonus = min(0.1, analytics.total_turns * 0.02)
            score = min(1.0, score + activity_bonus)

        return score

    def summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for the conversation."""
        analytics = self.analytics()
        return {
            "turns": analytics.total_turns,
            "success_rate": f"{analytics.success_rate:.2%}",
            "health_score": f"{self.health_score():.2f}",
            "avg_response_length": f"{analytics.avg_response_length:.0f} chars",
            "total_tokens": analytics.total_actual_tokens,
            "estimation_accuracy": (
                f"{analytics.estimation_accuracy:.2f}x"
                if analytics.estimation_accuracy
                else "N/A"
            ),
            "cache_enabled": self._state.cache is not None,
            "hints_enabled": self._state.hints is not None,
        }

    # --- Conversation branching and exploration ---
    def fork(self) -> Conversation:
        """Create a new conversation branch from current state.

        Returns a new Conversation with identical state but independent future.
        Useful for exploring alternative conversation paths.
        """
        return Conversation(self._state, self._executor)

    def rollback(self, to_turn: int = -1) -> Conversation:
        """Create a new conversation by rolling back to a previous turn.

        Args:
            to_turn: Turn index to roll back to. Negative values count from end.
                    -1 = remove last turn, -2 = remove last 2 turns, etc.
        """
        turns = list(self._state.turns)

        if not turns:
            return self  # Nothing to rollback

        if to_turn < 0:
            # Negative indexing: remove last N turns
            # to_turn = -1 means remove 1 turn, so keep (len - 1)
            # to_turn = -2 means remove 2 turns, so keep (len - 2)
            keep_count = len(turns) + to_turn
            keep_count = max(0, keep_count)  # Don't go below 0
        else:
            # Positive indexing: keep first N turns
            keep_count = min(to_turn + 1, len(turns))

        new_turns = tuple(turns[:keep_count])
        new_state = replace(
            self._state,
            turns=new_turns,
            version=self._state.version + 1,
        )
        return Conversation(new_state, self._executor)

    def branch_and_ask(self) -> tuple[Conversation, Conversation]:
        """Fork the conversation for future ask operations.

        Returns:
            tuple: (original_conversation, new_branch_for_ask)
        """
        branch = self.fork()
        # The ask operation is async and must be performed by the caller
        return self, branch

    async def explore_alternatives(
        self,
        prompts: Iterable[str],
    ) -> tuple[Conversation, dict[str, tuple[Conversation, str]]]:
        """Explore multiple alternative prompts from current conversation state.

        Args:
            prompts: Alternative prompts to try

        Returns:
            tuple: (original_conversation, {prompt: (branch_after_ask, response)})
        """
        alternatives: dict[str, tuple[Conversation, str]] = {}

        for prompt in prompts:
            branch = self.fork()
            branch_after = await branch.ask(prompt)
            response = (
                branch_after.state.last.assistant if branch_after.state.last else ""
            )
            alternatives[prompt] = (branch_after, response)

        return self, alternatives

    def merge_turns(
        self, other: Conversation, strategy: str = "chronological"
    ) -> Conversation:
        """Merge turns from another conversation branch.

        Args:
            other: Another conversation to merge turns from
            strategy: Merge strategy - "chronological", "append", or "interleave"
        """
        if strategy == "append":
            # Simply append all turns from other conversation
            merged_turns = (*self._state.turns, *other._state.turns)
        elif strategy == "chronological":
            # For simplicity, just append (could sort by timestamp if available)
            merged_turns = (*self._state.turns, *other._state.turns)
        elif strategy == "interleave":
            # Alternate turns between conversations
            self_turns = list(self._state.turns)
            other_turns = list(other._state.turns)
            merged = []

            max_len = max(len(self_turns), len(other_turns))
            for i in range(max_len):
                if i < len(self_turns):
                    merged.append(self_turns[i])
                if i < len(other_turns):
                    merged.append(other_turns[i])
            merged_turns = tuple(merged)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

        new_state = replace(
            self._state,
            turns=merged_turns,
            version=max(self._state.version, other._state.version) + 1,
        )
        return Conversation(new_state, self._executor)

    # --- Advanced context management ---
    def with_sliding_window(self, window_size: int) -> Conversation:
        """Create a conversation with sliding window context management.

        Keeps only the last N turns, providing bounded memory usage.
        """
        if window_size <= 0:
            raise ValueError("Window size must be positive")

        turns = list(self._state.turns)
        if len(turns) <= window_size:
            return self  # No need to truncate

        windowed_turns = tuple(turns[-window_size:])
        new_state = replace(
            self._state,
            turns=windowed_turns,
            version=self._state.version + 1,
        )
        return Conversation(new_state, self._executor)

    async def with_summarized_context(
        self,
        max_turns_to_keep: int = 3,
        summary_prompt: str = "Summarize the key points from this conversation so far:",
    ) -> Conversation:
        """Create a conversation with summarized earlier context.

        Replaces older turns with a summary, keeping recent turns intact.
        """
        turns = list(self._state.turns)

        if len(turns) <= max_turns_to_keep:
            return self  # Nothing to summarize

        # Split turns into "to summarize" and "to keep"
        turns_to_summarize = turns[:-max_turns_to_keep]
        turns_to_keep = turns[-max_turns_to_keep:]

        # Create a temporary conversation from turns to summarize
        summary_state = ConversationState(
            sources=self._state.sources,
            turns=tuple(turns_to_summarize),
            cache=self._state.cache,
            hints=self._state.hints,
            version=0,
        )
        temp_conv = Conversation(summary_state, self._executor)

        # Generate summary
        summary_conv = await temp_conv.ask(summary_prompt)
        summary_text = (
            summary_conv.state.last.assistant
            if summary_conv.state.last
            else "No summary available."
        )

        # Create synthetic summary exchange
        summary_exchange = Exchange(
            user="[Context Summary]",
            assistant=summary_text,
            error=False,
        )

        # Combine summary with recent turns
        new_turns = (summary_exchange, *turns_to_keep)
        new_state = replace(
            self._state,
            turns=new_turns,
            version=self._state.version + 1,
        )
        return Conversation(new_state, self._executor)

    def prune_context(
        self,
        strategy: str = "keep_important",
        target_turns: int = 10,
    ) -> Conversation:
        """Intelligently prune conversation context.

        Args:
            strategy: Pruning strategy - "keep_important", "keep_recent", "remove_errors"
            target_turns: Target number of turns to keep
        """
        turns = list(self._state.turns)

        if len(turns) <= target_turns:
            return self  # Nothing to prune

        if strategy == "keep_recent":
            pruned_turns = turns[-target_turns:]
        elif strategy == "remove_errors":
            # Remove error turns first, then apply recency if needed
            non_error_turns = [turn for turn in turns if not turn.error]
            if len(non_error_turns) <= target_turns:
                pruned_turns = non_error_turns
            else:
                pruned_turns = non_error_turns[-target_turns:]
        elif strategy == "keep_important":
            # Heuristic: keep longer responses as they're likely more important
            scored_turns = [
                (i, turn, len(turn.assistant) + len(turn.user))
                for i, turn in enumerate(turns)
            ]
            # Sort by score (length) descending, then by recency (index) descending
            scored_turns.sort(key=lambda x: (x[2], x[0]), reverse=True)

            # Take top target_turns, then sort by original order
            selected = sorted(
                scored_turns[:target_turns],
                key=lambda x: x[0],  # original index
            )
            pruned_turns = [turn for _, turn, _ in selected]
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}")

        new_state = replace(
            self._state,
            turns=tuple(pruned_turns),
            version=self._state.version + 1,
        )
        return Conversation(new_state, self._executor)

    def context_size_estimate(self) -> dict[str, int]:
        """Estimate the context size in various metrics."""
        turns = self._state.turns

        total_chars = sum(len(turn.user) + len(turn.assistant) for turn in turns)
        # Rough token estimate (4 chars per token average)
        estimated_tokens = total_chars // 4

        return {
            "turns": len(turns),
            "total_characters": total_chars,
            "estimated_tokens": estimated_tokens,
            "user_chars": sum(len(turn.user) for turn in turns),
            "assistant_chars": sum(len(turn.assistant) for turn in turns),
        }

    # --- Error handling and recovery ---
    async def ask_with_retry(
        self,
        prompt: str,
        *,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        retry_on_error: bool = True,
    ) -> Conversation:
        """Ask with automatic retry on failures.

        Args:
            prompt: The question to ask
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
            retry_on_error: Whether to retry on error responses
        """
        import asyncio

        last_error: Exception | None = None
        delay = 1.0  # Initial delay in seconds

        for attempt in range(max_retries + 1):
            try:
                result = await self.ask(prompt)

                # Check if we got an error response and should retry
                if (
                    retry_on_error
                    and result.state.last
                    and result.state.last.error
                    and attempt < max_retries
                ):
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
                    continue

                return result

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
                    continue
                # Max retries exceeded, create error exchange
                error_exchange = Exchange(
                    user=prompt,
                    assistant=f"Error after {max_retries + 1} attempts: {e!s}",
                    error=True,
                )
                new_state = ConversationState(
                    sources=self._state.sources,
                    turns=(*self._state.turns, error_exchange),
                    cache=self._state.cache,
                    hints=self._state.hints,
                    version=self._state.version + 1,
                )
                return Conversation(new_state, self._executor)

        # Should never reach here, but defensive fallback
        error_exchange = Exchange(
            user=prompt,
            assistant=f"Unknown error: {str(last_error) if last_error else 'Unknown'}",
            error=True,
        )
        new_state = ConversationState(
            sources=self._state.sources,
            turns=(*self._state.turns, error_exchange),
            cache=self._state.cache,
            hints=self._state.hints,
            version=self._state.version + 1,
        )
        return Conversation(new_state, self._executor)

    async def ask_with_fallback(
        self,
        prompt: str,
        fallback_prompts: Iterable[str],
    ) -> Conversation:
        """Ask with fallback prompts if the primary fails."""
        try:
            result = await self.ask(prompt)
            if not (result.state.last and result.state.last.error):
                return result
        except Exception as e:
            logger.exception("Error asking with primary prompt: %s", e)
            # Try fallbacks

        # Try fallback prompts
        for fallback in fallback_prompts:
            try:
                result = await self.ask(fallback)
                if not (result.state.last and result.state.last.error):
                    return result
            except Exception as e:
                logger.exception("Error asking with fallback prompt: %s", e)
                continue

        # All prompts failed, return error state
        error_exchange = Exchange(
            user=prompt,
            assistant="All prompts failed (primary and fallbacks)",
            error=True,
        )
        new_state = ConversationState(
            sources=self._state.sources,
            turns=(*self._state.turns, error_exchange),
            cache=self._state.cache,
            hints=self._state.hints,
            version=self._state.version + 1,
        )
        return Conversation(new_state, self._executor)

    def recover_from_errors(self, strategy: str = "remove") -> Conversation:
        """Recover from error states in the conversation.

        Args:
            strategy: Recovery strategy - "remove", "mark", or "retry_prompt"
        """
        turns = list(self._state.turns)

        if strategy == "remove":
            # Remove all error turns
            clean_turns = [turn for turn in turns if not turn.error]
        elif strategy == "mark":
            # Keep errors but mark them clearly
            clean_turns = []
            for turn in turns:
                if turn.error:
                    marked_turn = Exchange(
                        user=turn.user,
                        assistant=f"[RECOVERED ERROR: {turn.assistant}]",
                        error=False,  # Mark as recovered
                        estimate_min=turn.estimate_min,
                        estimate_max=turn.estimate_max,
                        actual_tokens=turn.actual_tokens,
                        in_range=turn.in_range,
                    )
                    clean_turns.append(marked_turn)
                else:
                    clean_turns.append(turn)
        elif strategy == "retry_prompt":
            # Keep a record that retries are needed (async operation needed)
            # For now, just mark them for retry
            clean_turns = []
            for turn in turns:
                if turn.error:
                    retry_turn = Exchange(
                        user=f"[RETRY NEEDED: {turn.user}]",
                        assistant="[Error marked for retry]",
                        error=False,
                    )
                    clean_turns.append(retry_turn)
                else:
                    clean_turns.append(turn)
        else:
            raise ValueError(f"Unknown recovery strategy: {strategy}")

        new_state = replace(
            self._state,
            turns=tuple(clean_turns),
            version=self._state.version + 1,
        )
        return Conversation(new_state, self._executor)

    def validate_state(self) -> dict[str, Any]:
        """Validate conversation state and return diagnostic information."""
        issues = []
        warnings = []

        # Check for error turns
        error_count = sum(1 for turn in self._state.turns if turn.error)
        if error_count > 0:
            issues.append(f"{error_count} turns have errors")

        # Check for empty responses
        empty_responses = sum(
            1 for turn in self._state.turns if not turn.assistant.strip()
        )
        if empty_responses > 0:
            warnings.append(f"{empty_responses} turns have empty responses")

        # Check token estimation accuracy
        analytics = self.analytics()
        if analytics.estimation_accuracy and analytics.estimation_accuracy < 0.5:
            warnings.append("Poor token estimation accuracy (< 50%)")

        # Check for very long context
        context_size = self.context_size_estimate()
        if context_size["estimated_tokens"] > 100000:  # Arbitrary large number
            warnings.append("Very large context size may impact performance")

        return {
            "is_healthy": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "error_rate": error_count / len(self._state.turns)
            if self._state.turns
            else 0.0,
            "context_size": context_size,
            "analytics": analytics,
        }


# --- Batch and Flow (minimal metrics surfaces) ---


@dataclass(frozen=True)
class BatchMetrics:
    per_prompt: tuple[dict[str, int | float], ...]
    totals: dict[str, int | float]


@dataclass(frozen=True)
class ConversationAnalytics:
    """Comprehensive analytics for a conversation."""

    total_turns: int
    error_turns: int
    success_rate: float

    # Token metrics
    total_estimated_tokens: int | None = None
    total_actual_tokens: int | None = None
    estimation_accuracy: float | None = None  # actual/estimated ratio

    # Cost metrics (if available)
    estimated_cost_range: tuple[float, float] | None = None  # (min, max) in USD

    # Performance metrics
    avg_response_time_ms: float | None = None
    cache_hit_rate: float | None = None

    # Content metrics
    avg_response_length: float = 0.0
    total_user_chars: int = 0
    total_assistant_chars: int = 0


@dataclass(frozen=True)
class FlowMetrics:
    steps: tuple[dict[str, int | float], ...]
    totals: dict[str, int | float]


class Flow:
    def __init__(self) -> None:
        self._steps: list[str] = []

    def ask(self, prompt: str) -> Flow:
        self._steps.append(prompt)
        return self

    @property
    def steps(self) -> tuple[str, ...]:
        return tuple(self._steps)


def _metrics_from_result(res: dict[str, Any]) -> dict[str, int | float]:
    totals: dict[str, int | float] = {}
    usage = res.get("usage")
    if isinstance(usage, dict):
        # Normalize a few common keys when available
        for key in (
            "prompt_tokens",
            "candidates_token_count",
            "total_tokens",
            "total_token_count",
        ):
            if key in usage and isinstance(usage[key], int | float):
                totals[key] = usage[key]
    metrics = res.get("metrics")
    if isinstance(metrics, dict):
        durations = metrics.get("durations")
        if isinstance(durations, dict):
            # Sum of stage durations if needed could be derived by caller
            totals["duration_ms_total"] = sum(
                float(v) * 1000 if v < 10 else float(v)  # rough guard
                for v in durations.values()
                if isinstance(v, int | float)
            )
    return totals
