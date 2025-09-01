#!/usr/bin/env python3
"""🎯 Recipe: Analyze a Single Paper Quickly.

When you need to: Extract key insights from one document with a concise prompt.

Ingredients:
- One local file (PDF/TXT/PNG/MP4 supported by your model)
- `GEMINI_API_KEY` set in the environment

What you'll learn:
- Use `run_simple` for a one-off analysis
- Pass a `Source` from a file or text
- Inspect answer and basic usage metrics

Difficulty: ⭐
Time: ~5 minutes
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from cookbook.utils.data_paths import pick_file_by_ext, resolve_data_dir
from gemini_batch import types
from gemini_batch.frontdoor import run_simple


async def main_async(path: Path, prompt: str) -> None:
    src = types.Source.from_file(path)
    env = await run_simple(prompt, source=src, prefer_json=False)

    status = env.get("status", "ok")
    answers = env.get("answers", [])
    usage = env.get("usage", {})
    print(f"Status: {status}")
    if answers:
        print("\n📋 Answer (first 400 chars):\n")
        print(str(answers[0])[:400] + ("..." if len(str(answers[0])) > 400 else ""))
    if isinstance(usage, dict):
        tok = usage.get("total_token_count")
        if tok is not None:
            print(f"\n🔢 Tokens: {tok}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a single paper")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a file to analyze",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Optional data directory override"
    )
    parser.add_argument(
        "--prompt",
        default="Summarize the key ideas and contributions in 5 bullets.",
        help="Prompt/question to ask",
    )
    args = parser.parse_args()

    file_path: Path
    if args.path is not None:
        file_path = args.path
    else:
        data_dir = resolve_data_dir(args.data_dir)
        pick = (
            pick_file_by_ext(data_dir, [".pdf", ".txt", ".png", ".jpg", ".mp4"]) or None
        )
        if pick is None:
            raise SystemExit(f"No suitable files found under {data_dir}")
        file_path = pick

    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")
    asyncio.run(main_async(file_path, args.prompt))


if __name__ == "__main__":
    main()
