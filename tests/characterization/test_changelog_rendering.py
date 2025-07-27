"""
Characterization test for Jinja2 changelog template rendering.
"""

import pytest


@pytest.mark.golden_test("golden_files/test_changelog_rendering.yml")
def test_changelog_template_renders_correctly(
    golden, jinja_env, mock_changelog_context
):
    """
    Characterizes the full output of the changelog Jinja template.

    This test avoids running the semantic-release CLI. Instead, it mocks
    the context data that is passed to the template and calls the rendering
    engine directly. This provides a fast, reliable, and deterministic test
    that fully covers the template's logic, including:
    - Section generation (features, fixes, etc.)
    - Commit formatting (scope, hash, links)
    - Breaking change sections
    - Released and Unreleased versions
    """
    # Arrange
    template = jinja_env.get_template("CHANGELOG.md.j2")

    # Act
    # Render the template using our mock context
    actual_changelog = template.render(context=mock_changelog_context)

    # Assert
    # The golden file contains the complete, expected changelog string
    assert actual_changelog == golden.out["expected_changelog"]
