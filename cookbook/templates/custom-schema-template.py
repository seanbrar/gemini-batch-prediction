#!/usr/bin/env python3
"""🎯 Template: Schema-First Extraction.

When you need to: Ask for structured JSON and parse into a Pydantic model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from pydantic import BaseModel

from cookbook.utils.data_paths import pick_file_by_ext, resolve_data_dir
from gemini_batch import types
from gemini_batch.frontdoor import run_batch


class MySchema(BaseModel):
    title: str
    items: list[str]


PROMPT = "Return JSON with keys: title (str), items (list[str])."


def _parse(answer: str) -> MySchema | None:
    try:
        return MySchema.model_validate(json.loads(answer))
    except Exception:
        return None


async def main_async(path: Path) -> None:
    src = types.Source.from_file(path)
    env = await run_batch([PROMPT], [src], prefer_json=True)
    ans = (env.get("answers") or [""])[0]
    data = _parse(str(ans))
    if data:
        print("✅ Parsed JSON successfully:")
        print(data)
    else:
        print("⚠️ Could not parse JSON (likely in mock mode). Raw response:")
        print(str(ans)[:400])


def main() -> None:
    parser = argparse.ArgumentParser(description="Schema-first extraction template")
    parser.add_argument("path", type=Path, nargs="?", default=None)
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Optional data directory override"
    )
    args = parser.parse_args()
    path = args.path
    if path is None:
        base = resolve_data_dir(args.data_dir)
        path = pick_file_by_ext(base, [".txt", ".md", ".pdf"]) or None
        if path is None:
            raise SystemExit(f"No suitable files found under {base}")
    asyncio.run(main_async(path))


if __name__ == "__main__":
    main()
