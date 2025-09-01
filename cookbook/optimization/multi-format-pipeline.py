#!/usr/bin/env python3
"""🎯 Recipe: Multi-Format Pipeline (PDF + Image + Video)

When you need to: Extract holistic insights from mixed media sources together
in one analysis pass.

Ingredients:
- A mix of PDF/TXT/PNG/MP4 files
- `GEMINI_API_KEY` set in the environment

What you'll learn:
- Build a mixed list of `Source` objects
- Ask layered prompts to combine insights
- Inspect results and usage

Difficulty: ⭐⭐⭐
Time: ~10-12 minutes
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from cookbook.utils.data_paths import pick_file_by_ext, resolve_data_dir
from gemini_batch import types
from gemini_batch.frontdoor import run_batch


async def main_async(
    pdf: Path | None, image: Path | None, video: Path | None, data_dir: Path | None
) -> None:
    if pdf is None or image is None or video is None:
        base = resolve_data_dir(data_dir)
        pdf = pdf or pick_file_by_ext(base, [".pdf", ".txt"]) or None
        image = image or pick_file_by_ext(base, [".png", ".jpg", ".jpeg"]) or None
        video = video or pick_file_by_ext(base, [".mp4", ".mov"]) or None
    # Guard against None before checking .exists()
    sources = [
        types.Source.from_file(p) for p in (pdf, image, video) if p and p.exists()
    ]
    if not sources:
        raise SystemExit("Provide at least one existing file among pdf/image/video")

    prompts = [
        "Summarize each source in 1-2 sentences.",
        "Synthesize: what common themes emerge across the media?",
        "List 3 actionable recommendations grounded in the content.",
    ]
    env = await run_batch(prompts, sources, prefer_json=False)
    print(f"Status: {env.get('status', 'ok')}")
    ans = env.get("answers", [])
    if ans:
        print("\n📋 Synthesis (first 500 chars):\n")
        print(str(ans[-1])[:500])


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-format analysis pipeline")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="PDF or text file",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Image file",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Video file",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Optional data directory override"
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.pdf, args.image, args.video, args.data_dir))


if __name__ == "__main__":
    main()
