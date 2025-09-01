#!/usr/bin/env python3
"""🎯 Recipe: Monitoring and Telemetry (Per-Stage Timings)

When you need to: Enable telemetry, inspect stage timings, and surface metrics
to plug into dashboards.

Ingredients:
- `GEMINI_BATCH_TELEMETRY=1` (recommended) or rely on envelope metrics
- A small batch (multiple prompts and at least one file)

What you'll learn:
- Enable telemetry via env and read per-stage durations
- Understand vectorization and parallel metrics
- Print a human-readable report

Difficulty: ⭐⭐⭐
Time: ~8-10 minutes
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from cookbook.utils.data_paths import resolve_data_dir
from gemini_batch import types
from gemini_batch.frontdoor import run_batch


def _print_durations(metrics: dict[str, Any]) -> None:
    durs = metrics.get("durations", {}) if isinstance(metrics, dict) else {}
    if not isinstance(durs, dict) or not durs:
        print("No stage durations available.")
        return
    print("\n⏱️  Stage durations (s):")
    for k, v in durs.items():
        try:
            print(f"  {k:<28} {float(v):.4f}")
        except Exception:
            continue


async def main_async(directory: Path) -> None:
    os.environ.setdefault("GEMINI_BATCH_TELEMETRY", "1")
    prompts = ["Identify three key takeaways.", "List top entities mentioned."]
    sources = types.sources_from_directory(directory)
    if not sources:
        raise SystemExit(f"No files found under {directory}")
    env = await run_batch(prompts, sources, prefer_json=False)
    print(f"Status: {env.get('status', 'ok')}")
    metrics = env.get("metrics", {})
    _print_durations(metrics)
    if isinstance(metrics, dict):
        if metrics.get("vectorized_n_calls") is not None:
            print(f"\n🔗 Vectorized API calls: {metrics.get('vectorized_n_calls')}")
        if metrics.get("parallel_n_calls") is not None:
            print(f"🧵 Parallel fan-out calls: {metrics.get('parallel_n_calls')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitoring and telemetry demo")
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=None,
        help="Directory with files to analyze",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Optional data directory override"
    )
    args = parser.parse_args()
    directory = args.directory or resolve_data_dir(args.data_dir)
    asyncio.run(main_async(directory))


if __name__ == "__main__":
    main()
