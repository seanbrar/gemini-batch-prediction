import copy
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
            "**(api)** Add new API endpoint ([`a1b2c3d`](https://github.com/USER/REPO/commit/dummy_hash))",
        ),
        (
            MockCommit(
                scope="",
                descriptions=["fix documentation typo"],
                short_hash="a1b2c3d",
                hexsha="dummy_hash",
            ),
            "Fix documentation typo ([`a1b2c3d`](https://github.com/USER/REPO/commit/dummy_hash))",
        ),
        (
            MockCommit(
                scope="",
                descriptions=["resolve integration issue"],
                linked_issues=["#123"],
                short_hash="a1b2c3d",
                hexsha="dummy_hash",
            ),
            "Resolve integration issue ([`a1b2c3d`](https://github.com/USER/REPO/commit/dummy_hash)) ([#123](https://github.com/USER/REPO/issues/123))",
        ),
        (
            MockCommit(
                scope="",
                descriptions=["handle multiple issues"],
                linked_issues=["#123", "#456"],
                short_hash="a1b2c3d",
                hexsha="dummy_hash",
            ),
            "Handle multiple issues ([`a1b2c3d`](https://github.com/USER/REPO/commit/dummy_hash)) ([#123](https://github.com/USER/REPO/issues/123), [#456](https://github.com/USER/REPO/issues/456))",
        ),
        (
            MockCommit(
                scope="llm",
                descriptions=["update the LLM prompt"],
                short_hash="a1b2c3d",
                hexsha="dummy_hash",
            ),
            "**(llm)** Update the LLM prompt ([`a1b2c3d`](https://github.com/USER/REPO/commit/dummy_hash))",
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
    "commits, expected_output",
    [
        (
            [
                MockCommit(
                    scope="", descriptions=["feat: A"], short_hash="h1", hexsha="d1"
                ),
                MockCommit(
                    scope="", descriptions=["fix: B"], short_hash="h2", hexsha="d2"
                ),
            ],
            "",
        ),
        (
            [
                MockCommit(
                    scope="",
                    descriptions=["feat: A"],
                    breaking_descriptions=["the API has changed"],
                    short_hash="h1",
                    hexsha="d1",
                )
            ],
            "### 💥 Breaking Changes\n\n- The API has changed",
        ),
        (
            [
                MockCommit(
                    scope="",
                    descriptions=["feat: A"],
                    breaking_descriptions=["the API has changed"],
                    short_hash="h1",
                    hexsha="d1",
                ),
                MockCommit(
                    scope="",
                    descriptions=["feat: B"],
                    breaking_descriptions=["Authentication is now required"],
                    short_hash="h2",
                    hexsha="d2",
                ),
            ],
            "### 💥 Breaking Changes\n\n- The API has changed\n- Authentication is now required",
        ),
        (
            [
                MockCommit(
                    scope="",
                    descriptions=["feat: A"],
                    breaking_descriptions=["the API has changed"],
                    short_hash="h1",
                    hexsha="d1",
                ),
                MockCommit(
                    scope="",
                    descriptions=["refactor: C"],
                    breaking_descriptions=["the API has changed"],
                    short_hash="h2",
                    hexsha="d2",
                ),
            ],
            "### 💥 Breaking Changes\n\n- The API has changed",
        ),
    ],
    ids=["no_breaking", "single_breaking", "multiple_breaking", "duplicate_breaking"],
)
def test_render_breaking_changes(macros, commits, expected_output):
    """Tests the `render_breaking_changes` macro directly."""
    actual_output = macros.render_breaking_changes(commits).strip()
    assert actual_output == expected_output


@pytest.mark.unit
def test_breaking_changes_section_rendering(jinja_env, mock_changelog_context):
    """
    Ensures the 'Breaking Changes' section is rendered correctly.
    """
    template = jinja_env.get_template("CHANGELOG.md.j2")

    # 1. Test with breaking changes present
    rendered_with_breaking = template.render(context=mock_changelog_context)
    assert "### 💥 Breaking Changes" in rendered_with_breaking

    # 2. Test with no breaking changes
    # Perform a deep copy for test isolation.
    safe_context = copy.deepcopy(mock_changelog_context)

    # Remove breaking changes from the "unreleased" section
    for commit_group in safe_context["history"]["unreleased"].values():
        for commit in commit_group:
            commit.breaking_descriptions = []

    # Remove breaking changes from all "released" sections
    for release_details in safe_context["history"]["released"].values():
        for commit_group in release_details["elements"].values():
            for commit in commit_group:
                commit.breaking_descriptions = []

    rendered_without_breaking = template.render(context=safe_context)
    assert "### 💥 Breaking Changes" not in rendered_without_breaking


@pytest.mark.unit
def test_render_release_sections(macros):
    """
    Tests that `render_release_sections` correctly renders a full release section.
    """
    commits_by_type = {
        "features": [
            MockCommit(
                scope="app",
                descriptions=["add new feature"],
                breaking_descriptions=["App is now different"],
                short_hash="h1",
                hexsha="d1",
            ),
        ],
        "bug fixes": [
            MockCommit(
                scope="api", descriptions=["fix a bug"], short_hash="h2", hexsha="d2"
            ),
        ],
        "documentation": [],  # Should not be rendered
    }

    actual_output = macros.render_release_sections(commits_by_type)

    # Check for breaking changes section
    assert "### 💥 Breaking Changes" in actual_output
    assert "- App is now different" in actual_output

    # Check for features section
    assert "### ✨ Features" in actual_output
    assert "**(app)** Add new feature" in actual_output

    # Check for bug fixes section
    assert "### 🐛 Bug Fixes" in actual_output
    assert "**(api)** Fix a bug" in actual_output

    # Check that empty sections are not rendered
    assert "### 📚 Documentation" not in actual_output


@pytest.mark.unit
@pytest.mark.parametrize(
    "commits_by_type, expected_result",
    [
        # No commits at all
        ({}, False),
        # Only empty sections
        ({"features": [], "bug fixes": [], "refactoring": []}, False),
        # Only non-renderable sections (chores, etc.)
        (
            {
                "chores": [
                    MockCommit(
                        type="chore",
                        scope="deps",
                        descriptions=["update deps"],
                        short_hash="abc",
                        hexsha="abc123",
                    )
                ]
            },
            False,
        ),
        # Mix of empty renderable and populated non-renderable
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
        # Has at least one renderable commit
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
            },
            True,
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
        "all_empty_sections",
        "only_chores",
        "empty_renderable_with_chores",
        "has_features",
        "mixed_renderable_and_chores",
    ],
)
def test_has_renderable_commits_macro(macros, commits_by_type, expected_result):
    """Tests the has_renderable_commits macro with various edge cases."""
    actual_result = macros.has_renderable_commits(commits_by_type)
    # Returns "true" or "" (empty string)
    actual_bool = bool(actual_result.strip())
    assert actual_bool == expected_result


@pytest.mark.unit
def test_unreleased_section_not_rendered_when_no_renderable_commits(jinja_env):
    """Test that Unreleased header is suppressed when there are no renderable commits."""
    # Context with only non-renderable commits
    context_with_chores_only = {
        "history": {
            "unreleased": {
                "chores": [
                    MockCommit(
                        type="chore",
                        scope="deps",
                        descriptions=["update dependencies"],
                        short_hash="abc123",
                        hexsha="abc123" * 5,
                    )
                ],
                # All renderable sections empty - but must be present as keys
                "features": [],
                "bug fixes": [],
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
    result = template.render(context=context_with_chores_only)

    # Should not contain Unreleased header or any section headers
    assert "## [Unreleased]" not in result
    assert "### ✨ Features" not in result
    assert "### 🐛 Bug Fixes" not in result


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
@pytest.mark.parametrize(
    "commit_type_to_test",
    [
        "chore",
        "test",
        "ci",
        "style",
        "security",
        "deps",
        "config",
        "docs",
        "wip",
        "temp",
        # Any other type not listed in the 'sections' macro
    ],
)
def test_breaking_change_on_filtered_commit_type(macros, commit_type_to_test):
    """
    Ensures breaking changes are rendered even if the commit's type
    (e.g., 'chore') is not configured to be displayed in the changelog.
    """
    # Arrange
    commits_by_type = {
        commit_type_to_test: [
            MockCommit(
                type=commit_type_to_test,
                scope="deps",
                descriptions=["drop support for old library"],
                breaking_descriptions=[
                    "Support for legacy library X has been removed."
                ],
                short_hash="h1",
                hexsha="d1",
            ),
        ]
    }

    # Act
    actual_output = macros.render_release_sections(commits_by_type)

    # Assert
    assert "### 💥 Breaking Changes" in actual_output
    assert "- Support for legacy library X has been removed." in actual_output
    # Verify the filtered section itself is NOT rendered
    assert f"### {commit_type_to_test.capitalize()}" not in actual_output


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
