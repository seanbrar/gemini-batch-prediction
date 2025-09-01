#!/usr/bin/env python3
"""🎯 Recipe: Comparative Analysis Across Sources

When you need to: Compare two or more documents side-by-side for similarities,
differences, strengths, and weaknesses.

Ingredients:
- 2+ files to compare
- `GEMINI_API_KEY` in environment

What you'll learn:
- Use a single prompt to drive a structured comparison
- Prefer JSON and parse defensively
- Print a concise diff summary

Difficulty: ⭐⭐⭐
Time: ~10 minutes
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path

from cookbook.utils.data_paths import resolve_data_dir
from gemini_batch import types
from gemini_batch.frontdoor import run_batch

PROMPT = (
    "Compare the provided sources and return JSON with keys: "
    "similarities (list), differences (list), strengths (list), weaknesses (list). "
    "Be specific and concise."
)


@dataclass
class Comparison:
    similarities: list[str]
    differences: list[str]
    strengths: list[str]
    weaknesses: list[str]


def _parse(answer: str) -> Comparison | None:
    try:
        data = json.loads(answer)
        return Comparison(
            similarities=list(map(str, data.get("similarities", []))),
            differences=list(map(str, data.get("differences", []))),
            strengths=list(map(str, data.get("strengths", []))),
            weaknesses=list(map(str, data.get("weaknesses", []))),
        )
    except Exception:
        return None


async def main_async(paths: list[Path]) -> None:
    srcs = [types.Source.from_file(p) for p in paths]
    env = await run_batch([PROMPT], srcs, prefer_json=True)
    ans = (env.get("answers") or [""])[0]
    comp = _parse(str(ans))
    if comp:
        print("\n✅ Comparison Summary")
        print(
            f"• Similarities: {len(comp.similarities)} | Differences: {len(comp.differences)}"
        )
        print(
            f"• Strengths: {len(comp.strengths)} | Weaknesses: {len(comp.weaknesses)}"
        )
        if comp.differences:
            print("\nFirst difference:")
            print(f"- {comp.differences[0]}")
    else:
        print("\n⚠️ Could not parse JSON; raw (first 400 chars):\n")
        print(str(ans)[:400])


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparative analysis across sources")
    parser.add_argument(
        "paths",
        type=Path,
        nargs="*",
        default=[],
        help="Two or more files to compare",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Optional data directory override"
    )
    args = parser.parse_args()
    paths: list[Path] = list(args.paths)
    if len(paths) < 2:
        data_dir = resolve_data_dir(args.data_dir)
        # Try to pick two files of common types
        exts = [".pdf", ".txt", ".jpg", ".png", ".md"]
        picks: list[Path] = []
        for p in sorted(data_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                picks.append(p)
                if len(picks) == 2:
                    break
        if len(picks) < 2:
            raise SystemExit(
                f"Need at least two files under {data_dir} or pass explicit paths"
            )
        paths = picks
    asyncio.run(main_async(paths))


if __name__ == "__main__":
    main()
