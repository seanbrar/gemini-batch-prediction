#!/usr/bin/env python3
"""🎯 Recipe: Custom Integrations via Telemetry Reporter

When you need to: Export timings/metrics to your system by attaching a custom
reporter (e.g., to logs, CSV, Prometheus, or a monitoring service).

Ingredients:
- Implement `TelemetryReporter` protocol methods
- Use `TelemetryContext` to enable and capture timings/metrics

What you'll learn:
- Write a minimal reporter
- Attach it and run a batch
- Inspect the collected data

Difficulty: ⭐⭐⭐
Time: ~8-12 minutes
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from cookbook.utils.data_paths import resolve_data_dir
from gemini_batch import types
from gemini_batch.frontdoor import run_batch
from gemini_batch.telemetry import TelemetryContext


class PrintReporter:
    def record_timing(self, scope: str, duration: float, **metadata: Any) -> None:
        print(f"TIMING {scope:<35} {duration:.4f}s depth={metadata.get('depth')}")

    def record_metric(self, scope: str, value: Any, **metadata: Any) -> None:
        print(f"METRIC {scope:<35} {value} parent={metadata.get('parent_scope')}")


async def main_async(directory: Path) -> None:
    prompts = ["List 3 key takeaways."]
    sources = types.sources_from_directory(directory)
    if not sources:
        raise SystemExit(f"No files found under {directory}")

    # Enable telemetry with our custom reporter for this scope
    ctx = TelemetryContext(PrintReporter())
    with ctx("cookbook.custom_integrations.run"):
        env = await run_batch(prompts, sources)
        print(f"Status: {env.get('status', 'ok')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom telemetry integration demo")
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=None,
        help="Directory with files",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Optional data directory override"
    )
    args = parser.parse_args()
    directory = args.directory or resolve_data_dir(args.data_dir)
    asyncio.run(main_async(directory))


if __name__ == "__main__":
    main()
