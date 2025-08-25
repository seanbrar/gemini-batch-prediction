from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from gemini_batch.core.types import InitialCommand

from .conversation_planner import compile_conversation
from .conversation_types import (
    BatchMetrics,
    ConversationAnalytics,
    ConversationPolicy,
    ConversationState,
    Exchange,
    PromptSet,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemini_batch.executor import GeminiExecutor


class Conversation:
    def __init__(self, state: ConversationState, executor: GeminiExecutor):
        self._state = state
        self._executor = executor

    @classmethod
    def start(
        cls, executor: GeminiExecutor, *, sources: Iterable[Any] = ()
    ) -> Conversation:
        return cls(ConversationState(sources=tuple(sources), turns=()), executor)

    @property
    def state(self) -> ConversationState:
        return self._state

    def with_policy(self, policy: ConversationPolicy) -> Conversation:
        return Conversation(
            replace(self._state, policy=policy, version=self._state.version + 1),
            self._executor,
        )

    def with_sources(self, sources: Iterable[Any]) -> Conversation:
        return Conversation(
            replace(
                self._state, sources=tuple(sources), version=self._state.version + 1
            ),
            self._executor,
        )

    async def ask(self, prompt: str) -> Conversation:
        ps = PromptSet((prompt,), "single")
        conv, answers, _ = await self.run(ps)
        return conv  # answers available at conv.state.last.assistant

    async def run(
        self, prompt_set: PromptSet
    ) -> tuple[Conversation, tuple[str, ...], BatchMetrics]:
        policy = self._state.policy
        plan = compile_conversation(self._state, prompt_set, policy)

        # Single pipeline seam: build InitialCommand and execute
        cfg = self._executor.config
        frozen = cfg.to_frozen() if hasattr(cfg, "to_frozen") else cfg
        cmd = InitialCommand(
            sources=plan.sources,
            prompts=plan.prompts,
            config=frozen,
            history=plan.history,
            hints=plan.hints,
        )
        res: dict[str, Any] = await self._executor.execute(
            cmd
        )  # core builds answers+metrics; do not reimplement

        # Map results minimally → Exchanges (no parsing/validation logic here)
        is_error = not bool(res.get("success", False))
        answers = tuple(str(a) for a in (res.get("answers") or []))
        usage = res.get("usage") or {}
        metrics = res.get("metrics") or {}
        token_val = metrics.get("token_validation") or {}

        # Extract validation warnings and surface them
        validation_warnings = res.get("validation_warnings") or ()
        if isinstance(validation_warnings, str):
            validation_warnings = (validation_warnings,)
        elif not isinstance(validation_warnings, list | tuple):
            validation_warnings = ()
        else:
            validation_warnings = tuple(str(w) for w in validation_warnings)

        # Add token estimation accuracy warnings for large mismatches
        warnings_list = list(validation_warnings)
        actual_tokens = token_val.get("actual")
        estimated_max = token_val.get("estimated_max")
        in_range = token_val.get("in_range")

        if actual_tokens and estimated_max and in_range is False:
            ratio = actual_tokens / estimated_max
            if ratio > 2.0:  # More than 2x over estimate
                warnings_list.append(
                    f"Token usage {actual_tokens} significantly exceeded estimate {estimated_max} ({ratio:.1f}x)"
                )

        validation_warnings = tuple(warnings_list)

        # For vec: append one synthetic batch Exchange with joined assistant text (audit-friendly, small)
        ex_text = answers[0] if prompt_set.mode == "single" else "; ".join(answers)
        ex = Exchange(
            user=f"[{prompt_set.mode} x{len(prompt_set.prompts)}]"
            if prompt_set.mode != "single"
            else prompt_set.prompts[0],
            assistant=ex_text,
            error=is_error,
            estimate_min=token_val.get("estimated_min"),
            estimate_max=token_val.get("estimated_max"),
            actual_tokens=token_val.get("actual")
            or usage.get("total_tokens")
            or usage.get("total_token_count"),
            in_range=token_val.get("in_range"),
            warnings=validation_warnings,
        )
        new_state = replace(
            self._state, turns=(*self._state.turns, ex), version=self._state.version + 1
        )

        # Improved per-prompt metrics handling
        # Try to extract per-prompt metrics from the metrics dict if available
        per_prompt_metrics = []
        if "per_prompt" in metrics and isinstance(metrics["per_prompt"], list | tuple):
            # Use actual per-prompt metrics if available
            per_prompt_metrics = list(metrics["per_prompt"])
        else:
            # Fall back to distributing totals across prompts
            base_metrics = dict(usage.items()) if isinstance(usage, dict) else {}
            if not base_metrics and isinstance(metrics, dict):
                # Try to use metrics dict as base if usage is empty
                base_metrics = {
                    k: v
                    for k, v in metrics.items()
                    if isinstance(v, int | float) and k != "token_validation"
                }
            per_prompt_metrics = [dict(base_metrics) for _ in answers]

        # Ensure we have the right number of per-prompt entries
        while len(per_prompt_metrics) < len(answers):
            per_prompt_metrics.append({})
        per_prompt = tuple(per_prompt_metrics[: len(answers)])

        # Build totals from usage or metrics
        totals = dict(usage.items()) if isinstance(usage, dict) else {}
        if not totals and isinstance(metrics, dict):
            totals = {
                k: v
                for k, v in metrics.items()
                if isinstance(v, int | float) and k != "token_validation"
            }

        return (
            Conversation(new_state, self._executor),
            answers,
            BatchMetrics(per_prompt=per_prompt, totals=totals),
        )

    # tiny analytics helper (pure; or move to conversation_analytics.py)
    def analytics(self) -> ConversationAnalytics:
        turns = self._state.turns
        n = len(turns)
        errs = sum(t.error for t in turns)
        tot_est = (
            sum(t.estimate_max for t in turns if t.estimate_max is not None) or None
        )
        tot_act = (
            sum(t.actual_tokens for t in turns if t.actual_tokens is not None) or None
        )
        acc = (tot_act / tot_est) if (tot_act and tot_est and tot_est > 0) else None
        total_user = sum(len(t.user) for t in turns)
        total_assist = sum(len(t.assistant) for t in turns)
        avg_len = (total_assist / n) if n else 0.0
        return ConversationAnalytics(
            n,
            errs,
            (n - errs) / n if n else 1.0,
            tot_est,
            tot_act,
            acc,
            avg_len,
            total_user,
            total_assist,
        )
