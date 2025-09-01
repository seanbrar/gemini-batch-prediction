from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


def test_tests_follow_taxonomy_layout() -> None:
    """All test modules reside under an approved type folder."""
    allowed = {
        "unit",
        "contracts",
        "integration",
        "workflows",
        "characterization",
        "performance",
    }
    root = Path("tests")
    violations: list[str] = []
    for path in root.rglob("test_*.py"):
        try:
            first = path.relative_to(root).parts[0]
        except Exception:  # pragma: no cover - defensive
            first = ""
        if first not in allowed:
            # Exempt top-level characterization legacy golden helpers if any in future, none today
            violations.append(str(path))

    assert not violations, (
        "Test files must live under one of {" + ", ".join(sorted(allowed)) + "}. "
        f"Found misplaced tests: {violations}"
    )
