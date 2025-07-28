"""Tests for GitHub Actions workflow configuration and contracts."""

from pathlib import Path

import pytest
import yaml


class TestWorkflowContracts:
    """Test that workflow calls match their contracts."""

    @pytest.mark.workflows
    def test_release_workflow_provides_required_inputs(self, reusable_checks_inputs):
        """Release workflow must provide all required inputs to reusable-checks."""
        with Path(".github/workflows/release.yml").open() as f:
            release = yaml.safe_load(f)

        pre_release_job = release["jobs"]["pre-release-checks"]
        provided_inputs = set(pre_release_job["with"].keys())

        missing = reusable_checks_inputs - provided_inputs
        assert not missing, f"Release workflow missing required inputs: {missing}"

    @pytest.mark.workflows
    def test_ci_workflow_provides_required_inputs(self, reusable_checks_inputs):
        """CI workflow must provide all required inputs to reusable-checks."""
        with Path(".github/workflows/ci.yml").open() as f:
            ci = yaml.safe_load(f)

        quality_job = ci["jobs"]["quality-checks"]
        provided_inputs = set(quality_job["with"].keys())

        missing = reusable_checks_inputs - provided_inputs
        assert not missing, f"CI workflow missing required inputs: {missing}"


class TestWorkflowSafety:
    """Test workflow safety configurations."""

    @pytest.mark.workflows
    def test_release_defaults_to_dry_run(self):
        """Release workflow should default to dry-run for safety."""
        with Path(".github/workflows/release.yml").open() as f:
            workflow = yaml.safe_load(f)

        dry_run_input = workflow["on"]["workflow_dispatch"]["inputs"]["dry_run"]
        assert dry_run_input["default"] is True, "Release should default to dry-run"

    @pytest.mark.workflows
    def test_release_has_concurrency_protection(self):
        """Release workflow should have concurrency protection."""
        with Path(".github/workflows/release.yml").open() as f:
            workflow = yaml.safe_load(f)

        release_job = workflow["jobs"]["release"]
        assert "concurrency" in release_job, (
            "Release job must have concurrency protection"
        )
        assert release_job["concurrency"]["cancel-in-progress"] is False


class TestSemanticReleaseConfig:
    """Test semantic-release configuration consistency."""

    @pytest.mark.workflows
    def test_main_branch_configuration(self, semantic_release_config):
        """Main branch should be configured correctly."""
        main_branch = semantic_release_config["branches"]["main"]

        assert main_branch["match"] == "main", (
            f"Expected 'main', got '{main_branch['match']}'"
        )
        assert main_branch["prerelease"] is False, (
            "Main branch should not be prerelease"
        )

    @pytest.mark.workflows
    def test_required_configuration_present(self, semantic_release_config):
        """Essential semantic-release configuration must be present."""
        required_keys = ["version_toml", "tag_format", "allow_zero_version"]

        for key in required_keys:
            assert key in semantic_release_config, f"Missing required config: {key}"

    @pytest.mark.workflows
    def test_version_toml_points_to_project_version(self, semantic_release_config):
        """version_toml should point to the correct location."""
        version_toml = semantic_release_config["version_toml"]
        assert "pyproject.toml:project.version" in version_toml
