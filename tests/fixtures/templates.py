"""Template and changelog-related fixtures.

Provides a configured Jinja environment and structured contexts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
import pytest

from tests.helpers import MockCommit


@pytest.fixture(scope="module")
def jinja_env():
    template_path = Path("templates")
    if not template_path.exists():
        pytest.skip("Changelog template directory not found.")

    env = Environment(  # noqa: S701
        loader=FileSystemLoader(searchpath=str(template_path)),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    env.filters["commit_hash_url"] = (
        lambda x: f"https://github.com/USER/REPO/commit/{x}"
    )
    env.filters["issue_url"] = (
        lambda x: f"https://github.com/USER/REPO/issues/{x.lstrip('#')}"
    )
    env.filters["compare_url"] = (
        lambda x, y: f"https://github.com/USER/REPO/compare/{x}...{y}"
    )

    return env


@pytest.fixture
def macros(jinja_env):
    return jinja_env.get_template(".macros.j2").module


@pytest.fixture
def mock_changelog_context():
    """Base context with unreleased changes and a single released version."""
    return {
        "history": {
            "unreleased": {
                "features": [
                    MockCommit(
                        type="feat",
                        scope="api",
                        descriptions=["add exciting new endpoint"],
                        short_hash="f34t001",
                        hexsha="f34t001" * 5,
                        breaking_descriptions=[],
                    ),
                ],
                "bug fixes": [
                    MockCommit(
                        type="fix",
                        scope="client",
                        descriptions=["standardize usage metadata key"],
                        short_hash="dd7b3e8",
                        hexsha="dd7b3e8" * 5,
                        breaking_descriptions=[],
                    ),
                ],
                "refactoring": [],
                "performance improvements": [],
                "reverts": [],
                "documentation": [],
                "build system": [],
            },
            "released": {
                "1.0.0": {
                    "version": "1.0.0",
                    "tagged_date": datetime(2024, 1, 1).date(),
                    "elements": {},
                }
            },
        }
    }


@pytest.fixture
def mock_changelog_context_no_breaking():
    return {
        "history": {
            "unreleased": {
                "bug fixes": [
                    MockCommit(
                        type="fix",
                        scope="client",
                        descriptions=["standardize usage metadata key"],
                        short_hash="dd7b3e8",
                        hexsha="dd7b3e8" * 5,
                        breaking_descriptions=[],
                    ),
                ],
                "features": [],
                "refactoring": [],
            },
            "released": {
                "1.0.0": {
                    "version": "1.0.0",
                    "tagged_date": datetime(2024, 7, 1).date(),
                    "elements": {
                        "features": [
                            MockCommit(
                                type="feat",
                                scope="core",
                                descriptions=["implement initial logic"],
                                short_hash="b4dfeat",
                                hexsha="b4dfeat" * 5,
                                breaking_descriptions=[
                                    "A breaking change related to implement initial logic."
                                ],
                            ),
                        ],
                        "documentation": [
                            MockCommit(
                                type="docs",
                                scope="",
                                descriptions=["add README and contributing guide"],
                                short_hash="d0c5001",
                                hexsha="d0c5001" * 5,
                            ),
                        ],
                        "build system": [
                            MockCommit(
                                type="build",
                                scope="",
                                descriptions=["configure pyproject.toml"],
                                short_hash="b51ld01",
                                hexsha="b51ld01" * 5,
                            ),
                        ],
                    },
                }
            },
        }
    }


@pytest.fixture
def mock_changelog_context_initial_release():
    return {
        "history": {
            "unreleased": {},
            "released": {
                "1.0.0": {
                    "version": "1.0.0",
                    "tagged_date": datetime(2024, 7, 1).date(),
                    "elements": {
                        "features": [
                            MockCommit(
                                type="feat",
                                scope="core",
                                descriptions=["implement initial logic"],
                                short_hash="b4dfeat",
                                hexsha="b4dfeat" * 5,
                                breaking_descriptions=[],
                            ),
                        ],
                        "documentation": [
                            MockCommit(
                                type="docs",
                                scope="",
                                descriptions=["add README and contributing guide"],
                                short_hash="d0c5001",
                                hexsha="d0c5001" * 5,
                            ),
                        ],
                    },
                }
            },
        }
    }


@pytest.fixture
def mock_changelog_context_multiple_releases():
    return {
        "history": {
            "unreleased": {
                "features": [
                    MockCommit(
                        type="feat",
                        scope="api",
                        descriptions=["add exciting new endpoint"],
                        short_hash="f34t001",
                        hexsha="f34t001" * 5,
                        breaking_descriptions=[],
                    ),
                ],
                "bug fixes": [],
                "refactoring": [],
                "performance improvements": [],
                "reverts": [],
                "documentation": [],
                "build system": [],
            },
            "released": {
                "1.1.0": {
                    "version": "1.1.0",
                    "tagged_date": datetime(2024, 8, 1).date(),
                    "elements": {
                        "features": [
                            MockCommit(
                                type="feat",
                                scope="client",
                                descriptions=["add client improvements"],
                                short_hash="f34t002",
                                hexsha="f34t002" * 5,
                                breaking_descriptions=[],
                            ),
                        ],
                        "bug fixes": [
                            MockCommit(
                                type="fix",
                                scope="api",
                                descriptions=["fix API response format"],
                                short_hash="f18x003",
                                hexsha="f18x003" * 5,
                            ),
                        ],
                    },
                },
                "1.0.0": {
                    "version": "1.0.0",
                    "tagged_date": datetime(2024, 7, 1).date(),
                    "elements": {
                        "features": [
                            MockCommit(
                                type="feat",
                                scope="core",
                                descriptions=["implement initial logic"],
                                short_hash="b4dfeat",
                                hexsha="b4dfeat" * 5,
                                breaking_descriptions=[],
                            ),
                        ],
                        "documentation": [
                            MockCommit(
                                type="docs",
                                scope="",
                                descriptions=["add README and contributing guide"],
                                short_hash="d0c5001",
                                hexsha="d0c5001" * 5,
                            ),
                        ],
                    },
                },
            },
        }
    }
