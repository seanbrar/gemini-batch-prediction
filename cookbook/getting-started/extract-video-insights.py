#!/usr/bin/env python3
"""🎯 Recipe: Extract Video Insights.

When you need to: Pull highlights, scenes, or key entities from a video file.

Ingredients:
- A small video file (MP4/MOV)
- `GEMINI_API_KEY` set in the environment

What you'll learn:
- Pass videos as `Source` objects
- Ask targeted questions to extract structure
- Print concise results with usage

Difficulty: ⭐⭐
Time: ~8 minutes
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from cookbook.utils.data_paths import pick_file_by_ext, resolve_data_dir
from gemini_batch import types
from gemini_batch.frontdoor import run_batch


def _first(env: dict[str, Any]) -> str:
    ans = env.get("answers", [])
    return str(ans[0]) if ans else ""


async def main_async(path: Path) -> None:
    src = types.Source.from_file(path)
    prompts = [
        "List 3 key moments with timestamps if visible.",
        "Identify main entities/objects and their roles.",
    ]
    env = await run_batch(prompts, [src], prefer_json=False)
    print(f"Status: {env.get('status', 'ok')}")
    print("\n🎞️  Highlights (first 400 chars):\n")
    print(_first(env)[:400])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract insights from a video")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a video",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Optional data directory override"
    )
    args = parser.parse_args()
    video_path: Path
    if args.path is None:
        data_dir = resolve_data_dir(args.data_dir)
        pick = pick_file_by_ext(data_dir, [".mp4", ".mov"]) or None
        if pick is None:
            raise SystemExit(f"No video files found under {data_dir}")
        video_path = pick
    else:
        video_path = args.path
    if not video_path.exists():
        raise SystemExit(f"File not found: {video_path}")
    asyncio.run(main_async(video_path))


if __name__ == "__main__":
    main()
