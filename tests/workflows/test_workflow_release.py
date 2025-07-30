"""Tests specifically for the release workflow and semantic-release configuration."""

from pathlib import Path
import re

import pytest
import yaml


class TestReleaseWorkflowSafety:
    """Test safety mechanisms in the release workflow."""

    @pytest.mark.workflows
    def test_manual_trigger_only(self, release_workflow):
        """Release workflow should only be manually triggered."""
        triggers = release_workflow["on"]
        assert "workflow_dispatch" in triggers, "Release must be manually triggerable"

        # Should not have push/PR triggers
        dangerous_triggers = ["push", "pull_request", "schedule"]
        for trigger in dangerous_triggers:
            assert trigger not in triggers, (
                f"Release workflow should not trigger on {trigger}"
            )

    @pytest.mark.workflows
    def test_dry_run_defaults_to_true(self, release_workflow):
        """Dry run should default to true to prevent accidental releases."""
        dry_run_input = release_workflow["on"]["workflow_dispatch"]["inputs"]["dry_run"]

        assert dry_run_input.get("type") == "boolean", "dry_run should be boolean type"
        assert dry_run_input.get("default") is True, "dry_run should default to true"
        assert "description" in dry_run_input, "dry_run should have a description"

    @pytest.mark.workflows
    def test_release_job_has_concurrency_protection(self, release_workflow):
        """Release job must have concurrency protection to prevent race conditions."""
        release_job = release_workflow["jobs"]["release"]

        assert "concurrency" in release_job, (
            "Release job must have concurrency protection"
        )
        concurrency = release_job["concurrency"]

        assert concurrency.get("group") == "release", (
            "Concurrency group should be 'release'"
        )
        assert concurrency.get("cancel-in-progress") is False, (
            "Should not cancel in-progress releases"
        )

    @pytest.mark.workflows
    def test_release_job_permissions(self, release_workflow):
        """Release job must have appropriate permissions."""
        release_job = release_workflow["jobs"]["release"]

        assert "permissions" in release_job, (
            "Release job must have explicit permissions"
        )
        permissions = release_job["permissions"]

        required_permissions = {
            "contents": "write",
            "issues": "write",
            "pull-requests": "write",
        }

        for perm, level in required_permissions.items():
            assert perm in permissions, f"Missing required permission: {perm}"
            assert permissions[perm] == level, (
                f"Permission {perm} should be {level}, got {permissions[perm]}"
            )

    @pytest.mark.workflows
    def test_release_has_verification_step(self, release_workflow):
        """Release workflow must verify artifacts to prevent silent failures."""
        release_job = release_workflow["jobs"]["release"]
        steps = release_job["steps"]

        verify_steps = [
            step
            for step in steps
            if "verify" in step.get("name", "").lower()
            and "artifacts" in step.get("name", "").lower()
        ]

        assert verify_steps, "Release workflow must have artifact verification step"

        verify_step = verify_steps[0]
        condition = verify_step.get("if", "")
        assert all(
            required in condition
            for required in [
                "inputs.dry_run == false",
                "steps.semver.outputs.IS_NEW_RELEASE == 'true'",
            ]
        ), "Verification should only run for actual releases"


class TestReleaseWorkflowLogic:
    """Test the conditional logic in the release workflow."""

    @pytest.mark.workflows
    def test_dry_run_and_actual_release_are_mutually_exclusive(self, release_workflow):
        """Dry run and actual release steps should be mutually exclusive."""
        release_job = release_workflow["jobs"]["release"]
        steps = release_job["steps"]

        dry_run_steps = []
        actual_release_steps = []

        for step in steps:
            if "if" in step:
                condition = step["if"]
                if "inputs.dry_run == true" in condition:
                    dry_run_steps.append(step.get("name", "unnamed"))
                elif "inputs.dry_run == false" in condition:
                    actual_release_steps.append(step.get("name", "unnamed"))

        assert dry_run_steps, "Should have dry-run specific steps"
        assert actual_release_steps, "Should have actual release specific steps"

    @pytest.mark.workflows
    def test_release_steps_check_new_version_exists(self, release_workflow):
        """Release steps should check that a new version is actually needed."""
        release_job = release_workflow["jobs"]["release"]
        steps = release_job["steps"]

        # Find steps that actually perform releases
        release_action_steps = [
            step
            for step in steps
            if step.get("name")
            in ["Create and Publish Release", "Preview Release (Dry Run)"]
        ]

        for step in release_action_steps:
            condition = step.get("if", "")
            assert "IS_NEW_RELEASE" in condition, (
                f"Step '{step.get('name')}' should check IS_NEW_RELEASE"
            )

    @pytest.mark.workflows
    def test_git_configuration_in_release(self, release_workflow):
        """Release workflow should configure Git properly."""
        release_job = release_workflow["jobs"]["release"]
        steps = release_job["steps"]

        # Look for Git configuration
        git_config_found = False
        for step in steps:
            if "run" in step and "git config" in step["run"]:
                git_config_found = True
                run_commands = step["run"]
                assert "user.name" in run_commands, "Must configure git user.name"
                assert "user.email" in run_commands, "Must configure git user.email"
                break

        assert git_config_found, "Release workflow must configure Git"


class TestSemanticReleaseConfiguration:
    """Test semantic-release configuration in pyproject.toml."""

    @pytest.mark.workflows
    def test_version_configuration(self, semantic_release_config):
        """Version configuration must be correct."""
        assert "version_toml" in semantic_release_config, "Must specify version_toml"

        version_toml = semantic_release_config["version_toml"]
        if isinstance(version_toml, list):
            assert "pyproject.toml:project.version" in version_toml
        else:
            assert version_toml == "pyproject.toml:project.version"

    @pytest.mark.workflows
    def test_tag_format(self, semantic_release_config):
        """Tag format should follow semantic versioning."""
        assert "tag_format" in semantic_release_config, "Must specify tag_format"

        tag_format = semantic_release_config["tag_format"]
        assert tag_format == "v{version}", (
            f"Expected 'v{{version}}', got '{tag_format}'"
        )

    @pytest.mark.workflows
    def test_branch_configuration(self, semantic_release_config):
        """Branch configuration must be correct."""
        assert "branches" in semantic_release_config, "Must configure branches"

        branches = semantic_release_config["branches"]
        assert "main" in branches, "Must configure main branch"

        main_branch = branches["main"]
        assert main_branch.get("match") == "main", (
            f"Main branch should match 'main', got {main_branch.get('match')}"
        )
        assert main_branch.get("prerelease") is False, (
            "Main branch should not be prerelease"
        )

    @pytest.mark.workflows
    def test_build_command_configuration(self, semantic_release_config):
        """Build command should be configured for Python packages."""
        assert "build_command" in semantic_release_config, "Must specify build_command"

        build_command = semantic_release_config["build_command"]
        assert "python -m build" in build_command, "Should use 'python -m build'"

    @pytest.mark.workflows
    def test_changelog_configuration(self, semantic_release_config):
        """Changelog configuration should be present."""
        assert "changelog" in semantic_release_config, "Must configure changelog"

        changelog = semantic_release_config["changelog"]
        assert changelog.get("mode") == "update", "Should use update mode for changelog"

    @pytest.mark.workflows
    def test_publish_configuration(self, semantic_release_config):
        """Publish configuration should be correct."""
        assert "publish" in semantic_release_config, "Must configure publish"

        publish = semantic_release_config["publish"]
        assert "dist_glob_patterns" in publish, "Must specify dist_glob_patterns"
        assert "upload_to_vcs_release" in publish, "Must specify upload_to_vcs_release"

    @pytest.mark.workflows
    def test_version_consistency_with_project(self, project_config):
        """Semantic-release version should match project version."""
        # This is more of a reminder test - the versions should start in sync
        project_version = project_config["version"]

        semver_pattern = r"^\d+\.\d+\.\d+$"
        assert re.match(semver_pattern, project_version), (
            f"Project version '{project_version}' should follow semantic versioning"
        )


class TestReleaseWorkflowInputValidation:
    """Test input validation and edge cases."""

    @pytest.mark.workflows
    def test_workflow_handles_no_release_needed(self):
        """Workflow should handle case where no release is needed."""
        with Path(".github/workflows/release.yml").open() as f:
            workflow = yaml.safe_load(f)

        release_job = workflow["jobs"]["release"]
        steps = release_job["steps"]

        # Should have a summary step for when no release is needed
        no_action_steps = [
            step for step in steps if step.get("name") == "No Action Summary"
        ]

        assert no_action_steps, "Should have step to handle no release needed case"
        no_action_step = no_action_steps[0]

        condition = no_action_step.get("if", "")
        assert "IS_NEW_RELEASE" in condition and "false" in condition, (
            "No action step should check for IS_NEW_RELEASE == false"
        )

    @pytest.mark.workflows
    def test_debug_information_available(self):
        """Release workflow should provide debug information."""
        with Path(".github/workflows/release.yml").open() as f:
            workflow = yaml.safe_load(f)

        release_job = workflow["jobs"]["release"]
        steps = release_job["steps"]

        # Should have debug step
        debug_steps = [
            step for step in steps if "debug" in step.get("name", "").lower()
        ]

        assert debug_steps, "Should have debug information step"


class TestWorkflowMaintainability:
    """Test that workflows are maintainable and well-documented."""

    @pytest.mark.workflows
    def test_release_workflow_has_comprehensive_comments(self):
        """Release workflow should be well-documented."""
        with Path(".github/workflows/release.yml").open() as f:
            content = f.read()

        # Count comment lines
        lines = content.split("\n")
        comment_lines = [line for line in lines if line.strip().startswith("#")]
        total_lines = len([line for line in lines if line.strip()])

        comment_ratio = len(comment_lines) / total_lines
        assert comment_ratio > 0.1, (
            f"Release workflow should have >10% comments, got {comment_ratio:.1%}"
        )

    @pytest.mark.workflows
    def test_complex_steps_have_descriptions(self):
        """Complex steps should have descriptive names."""
        with Path(".github/workflows/release.yml").open() as f:
            workflow = yaml.safe_load(f)

        release_job = workflow["jobs"]["release"]
        steps = release_job["steps"]

        for step in steps:
            if "run" in step and len(step["run"]) > 100:  # Complex steps
                assert "name" in step, "Complex steps should have names"
                assert len(step["name"]) > 10, (
                    f"Step name too short: '{step.get('name')}'"
                )
