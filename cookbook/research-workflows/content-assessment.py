#!/usr/bin/env python3
"""🎯 Recipe: Content Assessment for Learning Objectives

When you need to: Evaluate course materials (slides, PDFs, notes) for clarity,
coverage, difficulty, and alignment to learning objectives.

Ingredients:
- A directory of course materials
- `GEMINI_API_KEY` in environment

What you'll learn:
- Create assessment criteria prompts
- Prefer JSON for structured rubrics
- Print improvement suggestions

Difficulty: ⭐⭐⭐
Time: ~10-12 minutes
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
    "Assess these materials and return JSON with keys: "
    "clarity (1-5), coverage (1-5), difficulty (1-5), alignment (1-5), "
    "strengths (list), improvements (list)."
)


@dataclass
class Assessment:
    clarity: int
    coverage: int
    difficulty: int
    alignment: int
    strengths: list[str]
    improvements: list[str]


def _parse(answer: str) -> Assessment | None:
    try:
        data = json.loads(answer)
        return Assessment(
            clarity=int(data.get("clarity", 0)),
            coverage=int(data.get("coverage", 0)),
            difficulty=int(data.get("difficulty", 0)),
            alignment=int(data.get("alignment", 0)),
            strengths=[str(s) for s in data.get("strengths", [])],
            improvements=[str(s) for s in data.get("improvements", [])],
        )
    except Exception:
        return None


async def main_async(directory: Path) -> None:
    srcs = types.sources_from_directory(directory)
    if not srcs:
        raise SystemExit(f"No files found under {directory}")
    env = await run_batch([PROMPT], srcs, prefer_json=True)
    ans = (env.get("answers") or [""])[0]
    a = _parse(str(ans))
    if a:
        print("\n✅ Assessment")
        print(
            f"Scores — clarity:{a.clarity} coverage:{a.coverage} difficulty:{a.difficulty} alignment:{a.alignment}"
        )
        if a.improvements:
            print("\nTop improvement suggestion:")
            print(f"- {a.improvements[0]}")
    else:
        print("\n⚠️ Could not parse JSON; raw (first 400 chars):\n")
        print(str(ans)[:400])


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess course/lecture content")
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=None,
        help="Directory of materials",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Optional data directory override"
    )
    args = parser.parse_args()
    directory = args.directory or resolve_data_dir(args.data_dir)
    asyncio.run(main_async(directory))


if __name__ == "__main__":
    main()
