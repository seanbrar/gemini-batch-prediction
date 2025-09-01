import logging
from pathlib import Path
import subprocess
from subprocess import CompletedProcess

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.workflows]


@pytest.mark.workflows
@pytest.mark.integration
@pytest.mark.skipif(
    not Path("templates").exists(), reason="Templates directory not found"
)
def test_changelog_for_feature_commit(initialized_git_repo: Path) -> None:
    """
    Given an initialized repo,
    When a 'feat' commit is added,
    Then the changelog should contain an appropriate section for features.
    """
    logger = logging.getLogger(__name__)
    repo_path = initialized_git_repo

    # --- Helper to run commands ---
    def run(command: list[str]) -> CompletedProcess[str]:
        return subprocess.run(  # noqa: S603  # The command is a hardcoded list of strings, not user input.
            command,
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )

    # Arrange: Add the specific commit needed for this test.
    # Use a filename that is not ignored by the repo's .gitignore
    (repo_path / "feature.md").write_text("a feature")
    run(["git", "add", "feature.md"])
    run(["git", "commit", "-m", "feat(api): Add an awesome new feature"])

    # Act
    run(["python", "-m", "semantic_release", "changelog"])

    # Assert
    changelog_content = (repo_path / "CHANGELOG.md").read_text()
    logger.info("Generated Changelog:\n%s", changelog_content)

    # This project uses custom templates under `templates/` that follow Keep a Changelog.
    # Be intentionally strict: feat → "### Added" and bullet includes the description.
    assert "### Added" in changelog_content
    assert "- Add an awesome new feature" in changelog_content
