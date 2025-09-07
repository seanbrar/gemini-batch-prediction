"""Lightweight helpers for robust JSON extraction from model output.

These functions are intentionally dependency-free and handle common formatting
quirks such as Markdown code fences and stray prose surrounding a JSON block.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def _extract_fenced_json(text: str) -> str | None:
    """Return content inside a ```json ... ``` or ``` ... ``` fence, if present."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1)
    return None


def _extract_balanced_span(text: str) -> str | None:
    """Best-effort extraction of a balanced JSON object or array from text.

    Scans for the first '{' or '[' and returns the shortest balanced span. This
    ignores braces/brackets inside string literals in a simple, pragmatic way.
    """
    s = text
    start_obj = s.find("{")
    start_arr = s.find("[")
    starts = [x for x in (start_obj, start_arr) if x != -1]
    if not starts:
        return None
    start = min(starts)
    open_char = s[start]
    close_char = "}" if open_char == "{" else "]"

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def coerce_json(answer: str) -> Any | None:
    """Parse JSON from model output with common cleanup strategies.

    Order of attempts:
    1) Strict `json.loads(answer)`
    2) Extract fenced block (```json ... ``` or ``` ... ```), then parse
    3) Find a balanced object/array span and parse
    """
    txt = answer.strip()
    try:
        return json.loads(txt)
    except Exception:
        pass

    fenced = _extract_fenced_json(txt)
    if fenced is not None:
        try:
            return json.loads(fenced)
        except Exception:
            pass

    span = _extract_balanced_span(txt)
    if span is not None:
        try:
            return json.loads(span)
        except Exception:
            return None
    return None
