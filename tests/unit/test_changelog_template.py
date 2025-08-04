import logging

import pytest

from tests.helpers import MockCommit


@pytest.mark.unit
@pytest.mark.parametrize(
    "commit_object, expected_string",
    [
        (
            MockCommit(
                scope="api",
                descriptions=["add new API endpoint"],
                short_hash="a1b2c3d",
                hexsha="dummy_hash",
            ),
            " Add new API endpoint ([`a1b2c3d`](https://github.com/USER/REPO/commit/dummy_hash))",
        ),
        (
            MockCommit(
                scope="",
                descriptions=["fix documentation typo"],
                short_hash="a1b2c3d",
                hexsha="dummy_hash",
            ),
            " Fix documentation typo ([`a1b2c3d`](https://github.com/USER/REPO/commit/dummy_hash))",
        ),
        (
            MockCommit(
                scope="",
                descriptions=["resolve integration issue"],
                linked_issues=["#123"],
                short_hash="a1b2c3d",
                hexsha="dummy_hash",
            ),
            " Resolve integration issue ([`a1b2c3d`](https://github.com/USER/REPO/commit/dummy_hash)) ([#123](https://github.com/USER/REPO/issues/123))",
        ),
        (
            MockCommit(
                scope="",
                descriptions=["handle multiple issues"],
                linked_issues=["#123", "#456"],
                short_hash="a1b2c3d",
                hexsha="dummy_hash",
            ),
            " Handle multiple issues ([`a1b2c3d`](https://github.com/USER/REPO/commit/dummy_hash)) ([#123](https://github.com/USER/REPO/issues/123), [#456](https://github.com/USER/REPO/issues/456))",
        ),
        (
            MockCommit(
                scope="llm",
                descriptions=["update the LLM prompt"],
                short_hash="a1b2c3d",
                hexsha="dummy_hash",
            ),
            " Update the LLM prompt ([`a1b2c3d`](https://github.com/USER/REPO/commit/dummy_hash))",
        ),
    ],
    ids=["with_scope", "no_scope", "single_issue", "multiple_issues", "acronym_casing"],
)
def test_render_commit_content(macros, commit_object, expected_string):
    """Tests the `render_commit_content` macro directly."""
    actual_string = macros.render_commit_content(commit_object)
    assert actual_string == expected_string


@pytest.mark.unit
@pytest.mark.parametrize(
    "commits_by_type, expected_result",
    [
        # No commits at all
        ({}, False),
        # Only empty renderable sections
        ({"features": [], "bug fixes": [], "documentation": []}, False),
        # Only non-renderable sections (these shouldn't reach template due to PSR filtering)
        (
            {
                "chores": [
                    MockCommit(
                        type="chore",
                        scope="deps",
                        descriptions=["update"],
                        short_hash="abc",
                        hexsha="abc123",
                    )
                ]
            },
            False,
        ),
        # Has renderable features
        (
            {
                "features": [
                    MockCommit(
                        type="feat",
                        scope="api",
                        descriptions=["add endpoint"],
                        short_hash="def",
                        hexsha="def456",
                    )
                ]
            },
            True,
        ),
        # Has renderable bug fixes
        (
            {
                "bug fixes": [
                    MockCommit(
                        type="fix",
                        scope="api",
                        descriptions=["fix bug"],
                        short_hash="fix123",
                        hexsha="fix456",
                    )
                ]
            },
            True,
        ),
        # Has renderable documentation
        (
            {
                "documentation": [
                    MockCommit(
                        type="docs",
                        scope="",
                        descriptions=["update docs"],
                        short_hash="doc123",
                        hexsha="doc456",
                    )
                ]
            },
            True,
        ),
        # Has renderable performance improvements
        (
            {
                "performance improvements": [
                    MockCommit(
                        type="perf",
                        scope="core",
                        descriptions=["optimize"],
                        short_hash="perf123",
                        hexsha="perf456",
                    )
                ]
            },
            True,
        ),
        # Mix of empty renderable and non-renderable
        (
            {
                "features": [],
                "chores": [
                    MockCommit(
                        type="chore",
                        scope="deps",
                        descriptions=["update"],
                        short_hash="abc",
                        hexsha="abc123",
                    )
                ],
            },
            False,
        ),
        # Mix with some renderable
        (
            {
                "features": [
                    MockCommit(
                        type="feat",
                        scope="api",
                        descriptions=["add endpoint"],
                        short_hash="def",
                        hexsha="def456",
                    )
                ],
                "bug fixes": [],
                "chores": [
                    MockCommit(
                        type="chore",
                        scope="deps",
                        descriptions=["update"],
                        short_hash="abc",
                        hexsha="abc123",
                    )
                ],
            },
            True,
        ),
    ],
    ids=[
        "empty",
        "all_empty_renderable_sections",
        "only_non_renderable",
        "has_features",
        "has_bug_fixes",
        "has_documentation",
        "has_performance_improvements",
        "empty_renderable_with_non_renderable",
        "mixed_renderable_and_non_renderable",
    ],
)
def test_has_renderable_commits_macro(macros, commits_by_type, expected_result):
    """Tests the has_renderable_commits macro with the new section mappings."""
    actual_result = macros.has_renderable_commits(commits_by_type)
    # Returns "true" or "" (empty string)
    actual_bool = bool(actual_result.strip())
    assert actual_bool == expected_result


@pytest.mark.unit
def test_render_changelog_entry_single_section(macros):
    """Test rendering a single section with multiple commits."""
    release_data = {
        "features": [
            MockCommit(
                scope="api",
                descriptions=["add new endpoint"],
                short_hash="feat1",
                hexsha="feat1" * 5,
            ),
            MockCommit(
                scope="client",
                descriptions=["add client feature"],
                short_hash="feat2",
                hexsha="feat2" * 5,
            ),
        ]
    }

    result = macros.render_changelog_entry(release_data)

    # Should have section header
    assert "### Added" in result
    # Should have both commits
    assert "Add new endpoint" in result
    assert "Add client feature" in result
    # Should have proper bullet formatting
    assert "- Add new endpoint" in result
    assert "- Add client feature" in result
    # Should have blank line after header (non-compact mode)
    lines = result.split("\n")
    header_idx = next(i for i, line in enumerate(lines) if line == "### Added")
    assert lines[header_idx + 1] == ""  # Blank line after header


@pytest.mark.unit
def test_render_changelog_entry_multiple_sections(macros):
    """Test rendering multiple sections in correct order."""
    release_data = {
        "bug fixes": [
            MockCommit(
                scope="api",
                descriptions=["fix critical bug"],
                short_hash="fix1",
                hexsha="fix1" * 5,
            ),
        ],
        "features": [
            MockCommit(
                scope="core",
                descriptions=["add feature"],
                short_hash="feat1",
                hexsha="feat1" * 5,
            ),
        ],
        "documentation": [
            MockCommit(
                scope="",
                descriptions=["update README"],
                short_hash="doc1",
                hexsha="doc1" * 5,
            ),
        ],
    }

    result = macros.render_changelog_entry(release_data)

    # Should have all three sections
    assert "### Added" in result
    assert "### Fixed" in result
    assert "### Changed" in result

    # Should be in correct order (Added, Fixed, Changed)
    added_pos = result.find("### Added")
    fixed_pos = result.find("### Fixed")
    changed_pos = result.find("### Changed")

    assert added_pos < fixed_pos < changed_pos

    # Should have proper spacing between sections
    lines = result.split("\n")
    _added_idx = next(i for i, line in enumerate(lines) if line == "### Added")
    fixed_idx = next(i for i, line in enumerate(lines) if line == "### Fixed")

    # Should have a blank line before the second section
    assert lines[fixed_idx - 1] == ""


@pytest.mark.unit
def test_render_changelog_entry_compact_mode(macros):
    """Test that compact mode removes blank lines after headers."""
    release_data = {
        "features": [
            MockCommit(
                scope="api",
                descriptions=["add feature"],
                short_hash="feat1",
                hexsha="feat1" * 5,
            ),
        ]
    }

    result = macros.render_changelog_entry(release_data, compact=True)

    lines = result.split("\n")
    # In compact mode, header and content should be on the same line
    header_line = next(line for line in lines if line.startswith("### Added"))
    assert (
        header_line
        == "### Added- Add feature ([`feat1`](https://github.com/USER/REPO/commit/feat1feat1feat1feat1feat1))"
    )


@pytest.mark.unit
def test_render_changelog_entry_empty_sections_not_rendered(macros):
    """Test that empty sections are not rendered."""
    release_data = {
        "features": [
            MockCommit(
                scope="api",
                descriptions=["add feature"],
                short_hash="feat1",
                hexsha="feat1" * 5,
            ),
        ],
        "bug fixes": [],  # Empty section
        "documentation": [],  # Empty section
    }

    result = macros.render_changelog_entry(release_data)

    # Should only have Added section
    assert "### Added" in result
    assert "### Fixed" not in result
    assert "### Changed" not in result


@pytest.mark.unit
def test_render_changelog_entry_commit_sorting(macros):
    """Test that commits within a section are sorted by scope."""
    release_data = {
        "features": [
            MockCommit(
                scope="zzz",
                descriptions=["last feature"],
                short_hash="feat3",
                hexsha="feat3" * 5,
            ),
            MockCommit(
                scope="aaa",
                descriptions=["first feature"],
                short_hash="feat1",
                hexsha="feat1" * 5,
            ),
            MockCommit(
                scope="mmm",
                descriptions=["middle feature"],
                short_hash="feat2",
                hexsha="feat2" * 5,
            ),
        ]
    }

    result = macros.render_changelog_entry(release_data)
    lines = result.split("\n")

    # Find commit lines (start with "- ")
    commit_lines = [line for line in lines if line.startswith("- ")]

    # Should be sorted by scope: aaa, mmm, zzz
    # Since scopes are not in output, we verify by the descriptions
    assert "First feature" in commit_lines[0]  # aaa scope
    assert "Middle feature" in commit_lines[1]  # mmm scope
    assert "Last feature" in commit_lines[2]  # zzz scope


@pytest.mark.unit
def test_render_changelog_entry_section_mapping(macros):
    """Test that commit types are correctly mapped to section titles."""
    release_data = {
        "performance improvements": [
            MockCommit(
                scope="core",
                descriptions=["optimize algorithm"],
                short_hash="perf1",
                hexsha="perf1" * 5,
            ),
        ],
        "documentation": [
            MockCommit(
                scope="api",
                descriptions=["update API docs"],
                short_hash="doc1",
                hexsha="doc1" * 5,
            ),
        ],
    }

    result = macros.render_changelog_entry(release_data)

    # Both should map to "Changed" section
    assert "### Changed" in result
    assert "Optimize algorithm" in result
    assert "Update API docs" in result
    # Should not have separate sections
    assert result.count("### Changed") == 1


@pytest.mark.unit
def test_render_changelog_entry_no_renderable_commits(macros):
    """Test that macro returns empty string when no renderable commits."""
    release_data: dict[str, list[MockCommit]] = {}

    result = macros.render_changelog_entry(release_data)

    # Should be empty or just whitespace
    assert result.strip() == ""


@pytest.mark.unit
def test_render_commit_content_issue_formatting_fixed(macros):
    """Test that the issue formatting fix works correctly."""
    commit_with_issues = MockCommit(
        scope="api",
        descriptions=["fix critical bug"],
        linked_issues=["#123", "#456"],
        short_hash="a1b2c3d",
        hexsha="dummy_hash",
    )

    result = macros.render_commit_content(commit_with_issues)

    # Should be all on one line with proper spacing
    expected = "Fix critical bug ([`a1b2c3d`](https://github.com/USER/REPO/commit/dummy_hash)) ([#123](https://github.com/USER/REPO/issues/123), [#456](https://github.com/USER/REPO/issues/456))"
    assert result.strip() == expected

    # Should not contain any internal newlines
    assert "\n" not in result.strip()


@pytest.mark.unit
def test_no_breaking_changes_spacing_regression(jinja_env):
    """Test that spacing is correct when there are no breaking changes."""
    context_no_breaking = {
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
                "features": [
                    MockCommit(
                        type="feat",
                        scope="api",
                        descriptions=["add new endpoint"],
                        short_hash="feat123",
                        hexsha="feat123" * 5,
                        breaking_descriptions=[],
                    ),
                ],
                # Include all other sections as empty lists
                "refactoring": [],
                "performance improvements": [],
                "reverts": [],
                "documentation": [],
                "build system": [],
            },
            "released": {},
        }
    }

    template = jinja_env.get_template("CHANGELOG.md.j2")
    result = template.render(context=context_no_breaking)

    lines = result.split("\n")

    # Debug: Log the actual output around Unreleased section
    unreleased_idx = next(
        i for i, line in enumerate(lines) if line == "## [Unreleased]"
    )
    logging.info("Lines around Unreleased (index %s):", unreleased_idx)
    for i in range(max(0, unreleased_idx - 2), min(len(lines), unreleased_idx + 8)):
        logging.info("%2d: '%s'", i, lines[i])

    # Find first section header after Unreleased
    first_section_idx: int | None = None
    for i in range(unreleased_idx + 1, len(lines)):
        if lines[i].startswith("### "):
            first_section_idx = i
            break

    # Should be exactly one blank line between Unreleased header and first section
    assert first_section_idx is not None, "No section header found after Unreleased"
    assert unreleased_idx is not None, "Unreleased header not found"
    assert first_section_idx - unreleased_idx == 2, (
        "Expected 2 lines between Unreleased and first section, got "
        f"{first_section_idx - unreleased_idx}"
    )


@pytest.mark.unit
def test_render_comparison_links_initial_release(
    macros, mock_changelog_context_initial_release
):
    """
    Tests that comparison links are rendered correctly for the very first release
    when there is nothing to compare against.
    """
    # Arrange
    # (A new fixture with only one released version and no 'unreleased' section)

    # Act
    actual_links = macros.render_comparison_links(
        mock_changelog_context_initial_release["history"]
    )

    # Assert
    expected_link = "[1.0.0]: https://github.com/USER/REPO/releases/tag/v1.0.0"
    assert actual_links.strip() == expected_link


@pytest.mark.unit
def test_render_comparison_links_with_unreleased(macros, mock_changelog_context):
    """
    Tests that links are generated for both released and unreleased versions.
    """
    # Act
    actual_links = macros.render_comparison_links(mock_changelog_context["history"])

    # Assert
    assert (
        "[Unreleased]: https://github.com/USER/REPO/compare/v1.0.0...HEAD"
        in actual_links
    )
    assert "[1.0.0]: https://github.com/USER/REPO/releases/tag/v1.0.0" in actual_links
    # Verify correct line breaks
    assert actual_links.count("\n") == 1


@pytest.mark.unit
def test_render_comparison_links_multiple_releases(
    macros, mock_changelog_context_multiple_releases
):
    """
    Tests that the 'compare' URL is correctly generated between two released versions.
    """
    # Arrange
    # (A new fixture with at least two released versions: 1.1.0 and 1.0.0)

    # Act
    actual_links = macros.render_comparison_links(
        mock_changelog_context_multiple_releases["history"]
    )

    # Assert
    # Check the link for the newest version
    assert (
        "[1.1.0]: https://github.com/USER/REPO/compare/v1.0.0...v1.1.0" in actual_links
    )
    # Check the link for the oldest version
    assert "[1.0.0]: https://github.com/USER/REPO/releases/tag/v1.0.0" in actual_links


@pytest.mark.unit
def test_changelog_no_breaking_changes(jinja_env, mock_changelog_context_no_breaking):
    """Test formatting when there are no breaking changes."""
    template = jinja_env.get_template("CHANGELOG.md.j2")
    result = template.render(context=mock_changelog_context_no_breaking)

    # Should not have extra blank lines
    lines = result.split("\n")
    unreleased_idx = next(
        i for i, line in enumerate(lines) if line == "## [Unreleased]"
    )
    next_content_idx = next(
        i
        for i, line in enumerate(lines[unreleased_idx + 1 :], unreleased_idx + 1)
        if line.strip()
    )

    # There should be exactly one blank line between header and first section
    assert next_content_idx - unreleased_idx == 2
