import copy
from datetime import datetime
from pathlib import Path

import pytest

PREVIOUS_CHANGELOG = """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

<!-- PSR-INSERT-FLAG -->
## [1.2.3] - 2024-06-30

### ✨ Features

- **(existing)** An old feature from a previous release.
"""


def test_changelog_renders_correctly_in_update_mode(
    jinja_env, mock_changelog_context, fs
):
    """
    Validates that the template correctly prepends new content and regenerates links in "update" mode.

    This test uses pyfakefs to create a virtual `CHANGELOG.md` file and
    mocks the context that python-semantic-release provides when `mode` is "update".
    It ensures that the new "Unreleased" section is inserted correctly without
    disturbing the existing content.
    """

    # Arrange
    # 1. Create a fake filesystem with a PREVIOUS_CHANGELOG that includes an
    #    outdated or minimal link section.
    PREVIOUS_CHANGELOG_WITH_LINKS = (
        PREVIOUS_CHANGELOG
        + """
---
PSR-LINKS-START
[1.2.3]: https://github.com/USER/REPO/releases/tag/v1.2.3
"""
    )
    fs.create_file("CHANGELOG.md", contents=PREVIOUS_CHANGELOG_WITH_LINKS)

    # 2. Configure pyfakefs to allow access to the real templates directory
    #    This is better than copying template files as it uses the real templates
    fs.add_real_directory("templates", read_only=True)

    # 3. Add custom filters to the Jinja environment for this test.
    #    This mock reads from the pyfakefs virtual filesystem.
    def _read_file_from_fs(path):
        with Path(path).open() as f:
            return f.read()

    def _lstrip_filter(text):
        return text.lstrip() if text else ""

    jinja_env.filters["read_file"] = _read_file_from_fs
    jinja_env.filters["lstrip"] = _lstrip_filter

    # 4. Prepare a deepcopy of the context, but add a new release to it
    #    so the link regeneration can be tested.
    update_context = copy.deepcopy(mock_changelog_context)
    update_context["history"]["released"]["1.2.3"] = {
        "version": "1.2.3",
        "tagged_date": datetime(2024, 6, 30).date(),
        "elements": {},
    }

    # 5. Add the specific keys that PSR provides in "update" mode
    update_context["changelog_mode"] = "update"
    update_context["prev_changelog_file"] = "CHANGELOG.md"
    update_context["changelog_insertion_flag"] = "<!-- PSR-INSERT-FLAG -->"

    template = jinja_env.get_template("CHANGELOG.md.j2")

    # Act
    actual_changelog = template.render(context=update_context)

    # Assert
    # Check that the new "Unreleased" content is present
    assert "## [Unreleased]" in actual_changelog
    assert "Add exciting new endpoint" in actual_changelog

    # Check that the old, existing content is still present
    assert "## [1.2.3] - 2024-06-30" in actual_changelog
    assert "An old feature from a previous release" in actual_changelog

    # Check that the new content was inserted *before* the old content
    unreleased_position = actual_changelog.find("## [Unreleased]")
    previous_version_position = actual_changelog.find("## [1.2.3]")

    assert 0 < unreleased_position < previous_version_position

    # Check that the new, regenerated link section is correct
    assert (
        "[Unreleased]: https://github.com/USER/REPO/compare/v1.2.3...HEAD"
        in actual_changelog
    )
    assert (
        "[1.2.3]: https://github.com/USER/REPO/compare/v1.0.0...v1.2.3"
        in actual_changelog
    )
    assert (
        "[1.0.0]: https://github.com/USER/REPO/releases/tag/v1.0.0" in actual_changelog
    )


@pytest.mark.unit
def test_update_mode_auto_detection_empty_file(jinja_env, mock_changelog_context, fs):
    """Test that update mode auto-detects empty files and uses init mode logic."""

    # Create empty changelog file
    fs.create_file("CHANGELOG.md", contents="")
    fs.add_real_directory("templates", read_only=True)

    # Add filters
    def _read_file_from_fs(path):
        with Path(path).open() as f:
            return f.read()

    jinja_env.filters["read_file"] = _read_file_from_fs

    # Prepare update mode context
    update_context = copy.deepcopy(mock_changelog_context)
    update_context["changelog_mode"] = "update"
    update_context["prev_changelog_file"] = "CHANGELOG.md"

    template = jinja_env.get_template("CHANGELOG.md.j2")
    result = template.render(context=update_context)

    # Should render like init mode - with full header
    assert "# Changelog" in result
    assert "All notable changes to this project" in result
    assert "<!-- PSR-INSERT-FLAG -->" in result
    assert "## [Unreleased]" in result


@pytest.mark.unit
def test_update_mode_auto_detection_no_insertion_flag(
    jinja_env, mock_changelog_context, fs
):
    """Test that update mode auto-detects files without insertion flag and uses init mode logic."""

    # Create changelog file without insertion flag
    malformed_changelog = """# Changelog

Some content but no insertion flag.

## [1.0.0] - 2024-01-01
- Some old feature
"""

    fs.create_file("CHANGELOG.md", contents=malformed_changelog)
    fs.add_real_directory("templates", read_only=True)

    # Add filters
    def _read_file_from_fs(path):
        with Path(path).open() as f:
            return f.read()

    jinja_env.filters["read_file"] = _read_file_from_fs

    # Prepare update mode context
    update_context = copy.deepcopy(mock_changelog_context)
    update_context["changelog_mode"] = "update"
    update_context["prev_changelog_file"] = "CHANGELOG.md"

    template = jinja_env.get_template("CHANGELOG.md.j2")
    result = template.render(context=update_context)

    # Should render like init mode, ignoring the malformed file
    assert "# Changelog" in result
    assert "<!-- PSR-INSERT-FLAG -->" in result
    # Should not contain the malformed content
    assert "Some content but no insertion flag" not in result
