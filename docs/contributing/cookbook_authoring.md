# Cookbook Authoring Guide

Write problem‑first, copy‑pasteable recipes that help users succeed quickly while showcasing best practices of `gemini_batch`.

## Goals

- Show real scenarios users face, not features in isolation.
- Provide runnable code with clear inputs/outputs and expected success checks.
- Teach progressively: starter → research workflows → production patterns.

## Structure a Recipe

Start each recipe with a short, specific problem statement and a quick summary of what the reader will achieve.

Template:

```python
"""🎯 Recipe: [Descriptive Problem Statement]

When you need to: [One sentence describing the scenario]

Ingredients:
- [Prereqs: API key, data, deps]

What you'll learn:
- [Key concept 1]
- [Key concept 2]

Difficulty: ⭐–⭐⭐⭐⭐⭐   Time: ~N minutes
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from gemini_batch import create_executor, types


class OutputSchema(BaseModel):
    ...


def main() -> None:
    # 1) Setup & prerequisites
    # 2) Quick version (copy‑paste)
    # 3) Custom version (show flexibility)
    # 4) Print/validate results
    ...


if __name__ == "__main__":
    main()
```

Checklist:

- Imports included, types clear, minimal dependencies.
- Explain why key choices are made (not what lines do).
- Show a success check (e.g., assert shape, print metrics).
- Avoid secrets or private data in examples; use repo fixtures where possible.

## Progressive Levels

- Starter (getting‑started/): one core concept, minimal setup, copy‑paste ready.
- Research workflows: multi‑step processing, structured outputs, multi‑source synthesis.
- Production patterns: retries, telemetry, caching, adapters, and error handling.

## Writing Style

- Lead with the problem: “When you need to …”.
- Keep code concise and idiomatic (Python 3.13, type hints).
- Prefer clear names and small functions; add comments for rationale.
- Show both quick and customizable paths when useful.

## Quality Bar

- Actionable: runnable end‑to‑end with clear expected output.
- Verifiable: includes checks so success is provable.
- Safe: redact secrets, call out cost/tier notes when relevant.
- Connected: link to related how‑tos, concepts, and reference APIs.

## Where Examples Live

- Code examples live under `cookbook/` in the repo, organized by scenario (e.g., `getting-started/`, `research-workflows/`, `production/`).
- This guide complements `docs/cookbook.md` which indexes public recipes and patterns.

## See Also

- How‑to → Logging (`docs/how-to/logging.md`) for diagnostics in recipes.
- Explanation → Telemetry Spec for measuring performance.
- Explanation → Token Counting & Estimation for cost‑aware planning.
