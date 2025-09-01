#!/usr/bin/env python3
"""🎯 Recipe: Synthesize Research Findings Across Multiple Papers

When you need to: Analyze several papers and generate a structured synthesis
with key methodologies, findings, gaps, and implications.

Ingredients:
- `GEMINI_API_KEY` in environment
- Directory of research papers (PDF or extracted text)

What you'll learn:
- Multi-source analysis with shared context
- Prompt design for structured synthesis
- Displaying results and basic efficiency preview

Difficulty: ⭐⭐⭐
Time: ~10-15 minutes
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from pydantic import BaseModel

from cookbook.utils.data_paths import resolve_data_dir
from gemini_batch import types
from gemini_batch.frontdoor import run_batch


class LiteratureSynthesis(BaseModel):
    executive_summary: str
    key_methodologies: list[str]
    main_findings: list[str]
    research_gaps: list[str]
    future_directions: list[str]
    practical_implications: list[str]


PROMPTS: list[str] = [
    "List key methodological approaches across these papers.",
    "List key findings and how they relate.",
    "Identify gaps in the current research.",
    "Suggest future research directions.",
    "Describe practical implications for practitioners.",
    "Write a concise executive summary synthesizing all findings.",
]


def _efficiency_preview(n_sources: int, n_prompts: int) -> None:
    trad = n_sources * n_prompts
    print("📊 SYNTHESIS SCOPE:")
    print(f"Sources: {n_sources} | Prompts: {n_prompts}")
    print(f"Traditional naive calls (per source x prompt): ~{trad}")
    print("Vectorized prompts reuse shared context for planning efficiency.")


def _parse_structured(ans: str) -> LiteratureSynthesis | None:
    try:
        data = json.loads(ans)
        return LiteratureSynthesis.model_validate(data)
    except Exception:
        return None


async def main_async(directory: Path) -> None:
    sources = types.sources_from_directory(directory)
    if not sources:
        raise SystemExit(f"No files found under {directory}")

    _efficiency_preview(len(sources), len(PROMPTS))
    env = await run_batch(PROMPTS, sources, prefer_json=True)

    ans = (env.get("answers") or [""])[-1]
    synth = _parse_structured(str(ans))
    if synth:
        print("\n✅ Literature synthesis complete!\n")
        print("📄 Executive Summary (first 300 chars):")
        print(
            synth.executive_summary[:300]
            + ("..." if len(synth.executive_summary) > 300 else "")
        )
        print(
            f"\n🔬 Key Methodologies: {len(synth.key_methodologies)} | 🎯 Findings: {len(synth.main_findings)}"
        )
        print(
            f"🔍 Gaps: {len(synth.research_gaps)} | 🚀 Future: {len(synth.future_directions)} | 💡 Implications: {len(synth.practical_implications)}"
        )
    else:
        print(
            "\n⚠️ Structured parse failed (likely in mock mode); showing raw text (first 400 chars):\n"
        )
        print(str(ans)[:400])


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize literature findings")
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=None,
        help="Directory with papers",
    )
    args = parser.parse_args()
    directory = args.directory or resolve_data_dir(None)
    asyncio.run(main_async(directory))


if __name__ == "__main__":
    main()
