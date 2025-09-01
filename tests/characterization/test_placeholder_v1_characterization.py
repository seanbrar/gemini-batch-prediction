"""Placeholder for the v1 characterization suite.

This keeps the `characterization` lane visible while the new
suite is being (re)introduced. Historical YAML artifacts are
quarantined under `tests/characterization/legacy/golden_files/`.
"""

import pytest

pytestmark = pytest.mark.characterization


@pytest.mark.skip(reason="v1 characterization suite pending; artifacts quarantined")
def test_v1_characterization_placeholder() -> None:
    """Skip until v1 characterization tests are added."""
