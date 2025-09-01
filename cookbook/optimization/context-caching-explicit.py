#!/usr/bin/env python3
"""🎯 Recipe: Explicit Context Caching and Token Savings

When you need to: Analyze large content repeatedly, explicitly create a cache
once, then reuse it to reduce total tokens and latency on follow-up runs.

Ingredients:
- Directory with sizable files (PDF/TXT)
- `GEMINI_API_KEY` set in the environment

What you'll learn:
- Apply `CacheOptions` + `CachePolicyHint` with a deterministic key
- Run once to warm, then reuse-only to show token savings
- Read `ResultEnvelope.metrics.per_call_meta.cache_applied` and totals

Difficulty: ⭐⭐⭐
Time: ~10-12 minutes
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from cookbook.utils.data_paths import resolve_data_dir
from gemini_batch import types
from gemini_batch.frontdoor import run_batch
from gemini_batch.types import CacheOptions, CachePolicyHint, make_execution_options


async def _run(prompts: list[str], sources, *, opts) -> dict[str, Any]:
    return await run_batch(prompts, sources, prefer_json=False, options=opts)


def _tok(env: dict[str, Any]) -> int:
    u = env.get("usage") or {}
    try:
        return int(u.get("total_token_count", 0) or 0)
    except Exception:
        return 0


def _cache_hits(env: dict[str, Any]) -> int:
    m = env.get("metrics") or {}
    per = m.get("per_call_meta") or ()
    hits = 0
    for item in per:
        if isinstance(item, dict) and item.get("cache_applied"):
            hits += 1
    return hits


async def main_async(directory: Path, cache_key: str) -> None:
    prompts = [
        "List 5 key findings with one-sentence rationale.",
        "Extract 3 actionable recommendations.",
    ]
    sources = types.sources_from_directory(directory)
    if not sources:
        raise SystemExit(f"No files found under {directory}")

    warm_opts = make_execution_options(
        cache=CacheOptions(
            deterministic_key=cache_key, ttl_seconds=7200, reuse_only=False
        ),
        cache_policy=CachePolicyHint(first_turn_only=True),
    )
    reuse_opts = make_execution_options(
        cache=CacheOptions(
            deterministic_key=cache_key, ttl_seconds=7200, reuse_only=True
        ),
        cache_policy=CachePolicyHint(first_turn_only=True),
    )

    print("🔧 Warming cache...")
    warm = await _run(prompts, sources, opts=warm_opts)
    print("🔁 Reusing cache...")
    reuse = await _run(prompts, sources, opts=reuse_opts)

    warm_tok = _tok(warm)
    reuse_tok = _tok(reuse)
    saved = warm_tok - reuse_tok
    hits = _cache_hits(reuse)
    print("\n📊 RESULTS")
    print(f"Warm tokens:  {warm_tok:,}")
    print(f"Reuse tokens: {reuse_tok:,}")
    print(
        f"Savings:      {saved:,} tokens ({(saved / warm_tok * 100) if warm_tok else 0:.1f}%)"
    )
    print(f"Cache hits observed in per-call meta: {hits}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit context caching demo")
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=None,
        help="Directory to analyze and cache context from",
    )
    parser.add_argument("--key", default="cookbook-explicit-cache-key")
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Optional data directory override"
    )
    args = parser.parse_args()
    directory = args.directory or resolve_data_dir(args.data_dir)
    asyncio.run(main_async(directory, args.key))


if __name__ == "__main__":
    main()
