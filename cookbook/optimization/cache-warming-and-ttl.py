#!/usr/bin/env python3
"""🎯 Recipe: Cache Warming with Deterministic Keys and TTL

When you need to: Pre-warm shared context caches and then reuse them to
reduce latency and tokens across repeated analyses.

Ingredients:
- Set `enable_caching=True` in config (env or override)
- Deterministic cache key for the shared context
- Representative prompts and a directory of files

What you'll learn:
- Apply `CacheOptions` (deterministic key, TTL, reuse-only)
- Compare warm vs reuse timings and token usage
- Understand first-turn-only and token floor policy knobs

Difficulty: ⭐⭐⭐
Time: ~10-12 minutes
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from gemini_batch import types
from gemini_batch.config import resolve_config
from gemini_batch.frontdoor import run_batch
from gemini_batch.types import CacheOptions, CachePolicyHint, make_execution_options


async def _run(
    prompts: list[str], sources: tuple[types.Source, ...], *, opts
) -> dict[str, Any]:
    env = await run_batch(prompts, sources, prefer_json=False, options=opts)
    return {
        "status": env.get("status", "ok"),
        "usage": env.get("usage", {}),
        "metrics": env.get("metrics", {}),
    }


def _tokens(usage: dict[str, Any]) -> int:
    try:
        return int((usage or {}).get("total_token_count", 0) or 0)
    except Exception:
        return 0


async def main_async(directory: Path, cache_key: str) -> None:
    cfg = resolve_config(overrides={"enable_caching": True})
    print(f"⚙️  Caching enabled: {cfg.enable_caching}")

    prompts = [
        "List 5 key concepts with one-sentence explanations.",
        "Extract three actionable recommendations.",
    ]
    sources = types.sources_from_directory(directory)
    if not sources:
        raise SystemExit(f"No files found under {directory}")

    # Warm: allow create
    warm_opts = make_execution_options(
        cache=CacheOptions(
            deterministic_key=cache_key, ttl_seconds=3600, reuse_only=False
        ),
        cache_policy=CachePolicyHint(first_turn_only=True),
    )
    warm = await _run(prompts, sources, opts=warm_opts)

    # Reuse: reuse-only true
    reuse_opts = make_execution_options(
        cache=CacheOptions(
            deterministic_key=cache_key, ttl_seconds=3600, reuse_only=True
        ),
        cache_policy=CachePolicyHint(first_turn_only=True),
    )
    reuse = await _run(prompts, sources, opts=reuse_opts)

    w_tok = _tokens(warm.get("usage", {}))
    r_tok = _tokens(reuse.get("usage", {}))
    print("\n📊 RESULTS")
    print(f"Warm run status: {warm['status']} | tokens: {w_tok:,}")
    print(f"Reuse run status: {reuse['status']} | tokens: {r_tok:,}")
    print("Note: Tokens/metrics depend on provider support and may vary.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache warming and TTL demo")
    parser.add_argument("directory", type=Path, help="Directory to cache context from")
    parser.add_argument(
        "--key",
        default="cookbook-cache-key",
        help="Deterministic cache key to use",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args.directory, args.key))


if __name__ == "__main__":
    main()
