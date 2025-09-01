#!/usr/bin/env python3
"""🎯 Recipe: Batch Process Multiple Files Efficiently.

When you need to: Run a few questions across a directory of files in one go.

Ingredients:
- A directory of files (PDF, text, image, audio, video)
- `GEMINI_API_KEY` set in the environment

What you'll learn:
- Build `Source` objects from a directory
- Vectorize prompts via `run_batch`
- Inspect answers and per-prompt metrics

Difficulty: ⭐⭐
Time: ~8 minutes
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from cookbook.utils.data_paths import resolve_data_dir
from gemini_batch import types
from gemini_batch.frontdoor import run_batch


def _print_summary(env: dict[str, Any]) -> None:
    answers = env.get("answers", [])
    metrics = env.get("metrics", {})
    usage = env.get("usage", {})
    print(f"Answers returned: {len(answers)}")
    if isinstance(usage, dict):
        tok = usage.get("total_token_count")
        if tok is not None:
            print(f"🔢 Total tokens: {tok}")
    if isinstance(metrics, dict) and metrics.get("per_prompt"):
        print("\n⏱️  Per-prompt snapshots:")
        for p in metrics["per_prompt"]:
            idx = p.get("index")
            dur = (p.get("durations") or {}).get("execute.total")
            print(f"  Prompt[{idx}] duration: {dur if dur is not None else 'N/A'}s")


async def main_async(directory: Path) -> None:
    prompts = [
        "List 3 key takeaways.",
        "Extract the main entities mentioned.",
    ]
    sources = types.sources_from_directory(directory)
    if not sources:
        raise SystemExit(f"No files found under {directory}")

    env = await run_batch(prompts, sources, prefer_json=False)
    print(f"Status: {env.get('status', 'ok')}")
    _print_summary(env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process multiple files")
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=None,
        help="Directory containing files",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Optional data directory override"
    )
    args = parser.parse_args()
    directory = args.directory or resolve_data_dir(args.data_dir)
    asyncio.run(main_async(directory))


if __name__ == "__main__":
    main()
