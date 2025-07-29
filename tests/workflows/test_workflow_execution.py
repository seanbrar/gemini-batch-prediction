from pathlib import Path
import subprocess
import typing

import pytest


@pytest.mark.integration
def test_ci_lint_job_executes(act_executable: str) -> None:
    """
    Test that the 'lint-workflows' job in ci.yml can execute successfully.
    This test runs the actual job, not a dry-run, to validate its steps.
    """
    # Target a specific job with the -j flag for a faster, more focused test
    result = subprocess.run(  # noqa: S603
        [act_executable, "pull_request", "-j", "lint-workflows"],
        capture_output=True,
        text=True,
        check=False,  # Don't raise exception on non-zero exit
    )

    # Assert that the command's stdout contains the success message from actionlint
    assert "Job succeeded" in result.stdout, (
        f"Expected actionlint success log. Stderr: {result.stderr}"
    )

    assert result.returncode == 0, (
        f"'lint-workflows' job failed. Stderr: {result.stderr}"
    )


@pytest.mark.integration
def test_release_pre_checks_job_runs_tests(act_executable: str) -> None:
    """
    Test that the 'pre-release-checks' job in release.yml runs 'make test'.

    This is a fast, targeted check to ensure the quality gate that runs before
    a release is properly configured to call the test suite.
    """
    result = subprocess.run(  # noqa: S603
        [
            act_executable,
            "workflow_dispatch",
            "--dryrun",
            "-v",
            "-j",
            "pre-release-checks",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert "make test" in result.stdout, (
        "The dry-run for 'pre-release-checks' should show 'make test' "
        f"as a planned step. This ensures the test quality gate is active. Stderr: {result.stderr}"
    )
    assert result.returncode == 0, (
        f"The 'pre-release-checks' job failed to plan. Stderr: {result.stderr}"
    )


@pytest.mark.integration
def test_release_job_executes_setup_steps(act_executable: str) -> None:
    """
    Test that the 'release' job can execute its initial setup steps live.

    This test runs the 'release' job without '--dryrun' to provide runtime
    validation of the checkout and dependency installation process, which can
    catch issues that a dry-run would miss. The test expects to fail on later
    steps that require secrets.
    """
    result = subprocess.run(  # noqa: S603
        [
            act_executable,
            "workflow_dispatch",
            "-j",
            "release",
            "-W",
            ".github/workflows/release.yml",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # We assert that a step *after* checkout and setup has started. This implicitly
    # confirms the initial steps were successful.
    assert "Run Main Debug Repository State" in result.stdout, (
        "The 'release' job failed before or during the initial setup steps. "
        "This often indicates a problem with actions/checkout or dependency installation. "
        f"Stderr: {result.stderr}"
    )


@pytest.mark.integration
def test_release_job_triggers_on_new_feature_commit(
    act_executable: str,
    e2e_git_repo: tuple[
        Path, typing.Callable[[list[str]], subprocess.CompletedProcess[str]]
    ],
) -> None:
    """
    Given a repository with a new 'feat' commit, test that the release
    workflow correctly identifies a new version and completes a dry-run.

    This end-to-end test uses 'act' to run the 'release' job from the
    release.yml workflow within a fully isolated and realistic Git repository,
    validating the core semantic-release logic.
    """
    repo_path, run_in_repo = e2e_git_repo

    # 1. ARRANGE: Create a new commit that will trigger a release
    (repo_path / "new_feature.txt").write_text("A new feature")
    run_in_repo(["git", "add", "new_feature.txt"])
    run_in_repo(["git", "commit", "-m", "feat: add amazing new feature"])

    # 2. ACT: Run the release workflow using 'act'
    result = subprocess.run(  # noqa: S603
        [
            act_executable,
            "workflow_dispatch",
            "-j",
            "release",
            "--input",
            "dry_run=true",
        ],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )

    # 3. ASSERT: Verify the workflow behaved as expected
    assert "## 🛡️ Dry Run Completed" in result.stdout, (
        "The workflow should have completed a dry run and produced a summary. "
        f"This indicates a failure in the release logic. Stderr: {result.stderr}"
    )
    assert "🏁  Job succeeded" in result.stdout, (
        "The 'act' command should report that the job succeeded. "
        f"This often points to an issue with the workflow steps. Stderr: {result.stderr}"
    )
