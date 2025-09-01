#!/usr/bin/env python3
"""🎯 Recipe: Large-Scale Batching with Bounded Concurrency

When you need to: Ask the same question over many sources with client-side
fan-out and a safe concurrency limit.

Ingredients:
- A directory with many files
- `GEMINI_API_KEY` in environment

What you'll learn:
- Use `run_parallel` to fan out per-source calls
- Bound concurrency to respect rate/throughput
- Inspect aggregate status and metrics

Difficulty: ⭐⭐⭐
Time: ~10-12 minutes
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from cookbook.utils.data_paths import resolve_data_dir
from gemini_batch import types
from gemini_batch.frontdoor import run_parallel


async def main_async(directory: Path, prompt: str, concurrency: int) -> None:
    sources = types.sources_from_directory(directory)
    if not sources:
        raise SystemExit(f"No files found under {directory}")

    env = await run_parallel(
        prompt, sources, prefer_json=False, concurrency=concurrency
    )
    print(f"Status: {env.get('status', 'ok')}")
    metrics = env.get("metrics", {})
    if isinstance(metrics, dict):
        print(f"Parallel calls: {metrics.get('parallel_n_calls')}")
        if metrics.get("parallel_errors"):
            print(f"Errors: {metrics.get('parallel_errors')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Large-scale batching with concurrency"
    )
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=None,
        help="Directory of files",
    )
    parser.add_argument(
        "--prompt",
        default="Extract three key takeaways.",
        help="Question to ask each source",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max client-side fan-out",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Optional data directory override"
    )
    args = parser.parse_args()
    directory = args.directory or resolve_data_dir(args.data_dir)
    asyncio.run(main_async(directory, args.prompt, args.concurrency))


if __name__ == "__main__":
    main()
