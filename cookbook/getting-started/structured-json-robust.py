#!/usr/bin/env python3
"""🎯 Recipe: Structured JSON With Robust Fallback Parsing

When you need to: Get well-structured JSON from a model, bias extraction to
JSON, and parse into a Pydantic schema with safe fallbacks while inspecting
the stable `ResultEnvelope` for metrics and usage.

Ingredients:
- One local file (report, article, or transcript as PDF/TXT)
- `GEMINI_API_KEY` set in the environment

What you'll learn:
- Prompt pattern + JSON bias (`prefer_json=True`) for structured results
- Robust fallback parsing of LLM output into a Pydantic schema
- Inspecting `ResultEnvelope` fields: `status`, `answers`, `usage`, `metrics`

Difficulty: ⭐⭐
Time: ~8-10 minutes
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel

from cookbook.utils.data_paths import pick_file_by_ext, resolve_data_dir
from gemini_batch import types
from gemini_batch.core.result_envelope import (
    explain_invalid_result_envelope,
    is_result_envelope,
)
from gemini_batch.frontdoor import run_batch


class KeyPoints(BaseModel):
    """Schema for structured analysis results."""

    title: str
    bullets: list[str]
    summary: str


PROMPT = (
    "Return application/json ONLY. Produce a compact JSON object with keys: "
    "title (short string), bullets (list of 3-5 concise strings), summary (3-4 sentences)."
)


def _extract_json_block(text: str) -> str | None:
    """Best-effort JSON block extractor.

    Looks for fenced blocks (```json ... ```), then fallback to first {...}
    span. Returns a candidate JSON string or None.
    """
    fence = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1)
    brace = re.search(r"\{[\s\S]*\}$", text.strip())  # greedy tail match
    if brace:
        return brace.group(0)
    return None


def _robust_parse(answer: str) -> KeyPoints | None:
    """Try strict JSON first, then a relaxed fenced/brace extraction."""
    try:
        return KeyPoints.model_validate(json.loads(answer))
    except Exception:
        pass
    candidate = _extract_json_block(answer)
    if candidate is None:
        return None
    try:
        return KeyPoints.model_validate(json.loads(candidate))
    except Exception:
        return None


def _print_envelope_debug(env: dict[str, Any]) -> None:
    print(f"Status: {env.get('status', 'ok')} | Method: {env.get('extraction_method')}")
    usage = env.get("usage") or {}
    total = usage.get("total_token_count") or usage.get("total_tokens")
    if total is not None:
        print(f"🔢 Total tokens: {total}")
    m = env.get("metrics") or {}
    hints = (m.get("hints") or {}) if isinstance(m, dict) else {}
    if hints:
        print(f"Hints: {hints}")
    durs = (m.get("durations") or {}) if isinstance(m, dict) else {}
    if durs:
        # show a couple of key stages if present
        api_t = durs.get("APIHandler.execute") or durs.get("APIHandler")
        plan_t = durs.get("ExecutionPlanner")
        if api_t is not None or plan_t is not None:
            print(
                "⏱️  Durations (s):",
                {
                    k: v
                    for k, v in {"plan": plan_t, "api": api_t}.items()
                    if v is not None
                },
            )


async def main_async(path: Path) -> None:
    src = types.Source.from_file(path)
    env = await run_batch([PROMPT], [src], prefer_json=True)

    # Validate and inspect the envelope
    if not is_result_envelope(env):
        reason = explain_invalid_result_envelope(env) or "invalid envelope"
        raise SystemExit(f"Unexpected result shape: {reason}")

    _print_envelope_debug(env)

    # Parse first answer
    answers = env.get("answers", [])
    raw = str(answers[0]) if answers else ""
    parsed = _robust_parse(raw)

    print("\n📋 Structured result:")
    if parsed:
        print(parsed.model_dump_json(indent=2))
    else:
        print("(fallback to raw)")
        print(raw[:600] + ("..." if len(raw) > 600 else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured JSON with robust parsing")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a PDF/TXT file",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Optional data directory override"
    )
    args = parser.parse_args()

    file_path: Path
    if args.path is not None:
        file_path = args.path
    else:
        data_dir = resolve_data_dir(args.data_dir)
        pick = pick_file_by_ext(data_dir, [".pdf", ".txt"]) or None
        if pick is None:
            raise SystemExit(f"No PDF/TXT files found under {data_dir}")
        file_path = pick

    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")
    asyncio.run(main_async(file_path))


if __name__ == "__main__":
    main()
