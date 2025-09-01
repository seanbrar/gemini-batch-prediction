#!/usr/bin/env python3
"""🎯 Recipe: Efficiency Comparison — Vectorized vs Naive

When you need to: Quantify token/time savings of vectorized prompts over a
shared context compared to a naive loop that calls once per prompt.

Ingredients:
- A directory of files or one file with multiple prompts
- `GEMINI_API_KEY` set in the environment

What you'll learn:
- Use `research.compare_efficiency` to benchmark both paths
- Interpret tokens/time/call ratios and basic environment capture
- Optional aggregate mode for single-call multi-answer JSON

Difficulty: ⭐⭐
Time: ~8-10 minutes
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from cookbook.utils.data_paths import resolve_data_dir
from gemini_batch import types
from gemini_batch.research import compare_efficiency


async def main_async(directory: Path, mode: str, trials: int) -> None:
    prompts = [
        "List 3 key takeaways.",
        "Extract top entities.",
        "Summarize in 3 bullets.",
    ]
    sources = types.sources_from_directory(directory)
    rep = await compare_efficiency(
        prompts,
        sources,
        prefer_json=(mode == "aggregate"),
        mode=mode if mode in ("batch", "aggregate") else "auto",
        trials=max(1, trials),
        warmup=1,
        include_pipeline_durations=True,
        label="cookbook-demo",
    )
    print("\n📊 Efficiency Summary:")
    print(rep.summary(verbose=(trials > 1)))
    # Optional: print environment and call counts
    print("Env:", rep.env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Efficiency comparison demo")
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=None,
        help="Directory with files to analyze",
    )
    parser.add_argument(
        "--mode",
        choices=["batch", "aggregate", "auto"],
        default="auto",
        help="Vectorized mode: multi-call batch vs single-call aggregate",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Optional data directory override"
    )
    args = parser.parse_args()
    directory = args.directory or resolve_data_dir(args.data_dir)
    asyncio.run(main_async(directory, args.mode, args.trials))


if __name__ == "__main__":
    main()
