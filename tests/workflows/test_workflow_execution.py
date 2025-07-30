"""
Integration tests for GitHub Actions workflows using act.

These tests validate that workflow files can execute correctly and produce
expected outputs in isolated environments.
"""

import pytest

from tests.helpers import ActTestHelper


@pytest.mark.integration
def test_ci_lint_job_executes(act_helper: ActTestHelper) -> None:
    """
    Test that the 'lint-workflows' job in ci.yml can execute successfully.
    This test runs the actual job, not a dry-run, to validate its steps.
    """
    result = act_helper.run_lint_workflow()
    result.assert_contains("Job succeeded").assert_success("Lint workflows job")


@pytest.mark.integration
def test_release_pre_checks_job_runs_tests(act_helper: ActTestHelper) -> None:
    """
    Test that the 'pre-release-checks' job in release.yml runs 'make test'.

    This is a fast, targeted check to ensure the quality gate that runs before
    a release is properly configured to call the test suite.
    """
    result = act_helper.run_pre_release_checks_dry_run()
    result.assert_contains(
        "make test", "Pre-release checks dry-run should show 'make test' step"
    ).assert_success("Pre-release checks dry-run")


@pytest.mark.integration
def test_release_job_executes_setup_steps(act_helper: ActTestHelper) -> None:
    """
    Test that the 'release' job can execute its initial setup steps live.

    This test runs the 'release' job without '--dryrun' to provide runtime
    validation of the checkout and dependency installation process, which can
    catch issues that a dry-run would miss. The test expects to fail on later
    steps that require secrets.
    """
    result = act_helper.run_release_workflow()

    # We assert that a step *after* checkout and setup has started. This implicitly
    # confirms the initial steps were successful.
    result.assert_contains(
        "Run Main Debug Repository State",
        "Release job should complete initial setup steps",
    )


@pytest.mark.integration
def test_release_job_triggers_on_new_feature_commit(act_helper: ActTestHelper) -> None:
    """
    Given a repository with a new 'feat' commit, test that the release
    workflow correctly identifies a new version and completes a dry-run.

    This end-to-end test uses 'act' to run the 'release' job from the
    release.yml workflow within a fully isolated and realistic Git repository,
    validating the core semantic-release logic.
    """
    # ARRANGE: Create a new commit that will trigger a release
    act_helper.git.create_feature_commit()

    # ACT: Run the release workflow in dry-run mode
    result = act_helper.run_release_workflow(dry_run=True)

    # ASSERT: Verify the workflow behaved as expected
    result.assert_contains(
        "## 🛡️ Dry Run Completed", "Workflow should complete dry run with summary"
    ).assert_contains("🏁  Job succeeded", "Act should report job success")
