import subprocess

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
def test_release_workflow_dry_run_previews_version(act_executable: str) -> None:
    """Test that the release dry-run correctly previews a release."""
    # We still use --dryrun here because a real release is complex
    result = subprocess.run(  # noqa: S603
        [
            act_executable,
            "workflow_dispatch",
            "--dryrun",
            "-v",
            "-j",
            "pre-release-checks",  # Target a specific job
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert "make test" in result.stdout, (
        f"Release dry-run should have run 'make test'. Stderr: {result.stderr}"
    )

    assert result.returncode == 0, (
        f"Release 'pre-release-checks' job failed. Stderr: {result.stderr}"
    )
