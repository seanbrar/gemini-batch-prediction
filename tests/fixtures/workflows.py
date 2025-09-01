"""Workflow and repository metadata fixtures.

Includes ephemeral git repo setup that mirrors the working tree so GitHub
Actions workflows executed via ``act`` see a realistic repository. Also
exposes helpers to read workflow and project configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tomllib
from typing import TYPE_CHECKING

import pytest
import yaml

from tests.helpers import ActTestHelper, GitHelper

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="session")
def act_executable() -> str:
    """Resolve ``act`` binary and ensure Docker is usable; otherwise skip.

    These workflow execution tests are integration/slow and depend on a working
    local Docker daemon. If ``act`` or Docker is not available, we skip early
    rather than failing with environment-specific errors.
    """
    path = shutil.which("act")
    if not path:
        pytest.skip("'act' not found in PATH; skipping workflow execution tests.")

    # Reduce noisy network calls for notices in environments without egress
    os.environ.setdefault("ACT_DISABLE_NOTICE", "true")

    # Require Docker CLI and a responsive daemon
    docker_bin = shutil.which("docker")
    if not docker_bin:
        pytest.skip("Docker CLI not found; skipping workflow execution tests.")

    # Common local daemon socket; still verify with `docker info`
    try:
        proc = subprocess.run(  # noqa: S603
            [docker_bin, "info"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # pragma: no cover - defensive, environment-specific
        pytest.skip("Docker daemon not reachable; skipping workflow execution tests.")

    if proc.returncode != 0:
        pytest.skip("Docker daemon not reachable; skipping workflow execution tests.")

    return path


@pytest.fixture
def initialized_git_repo(tmp_path: Path) -> Generator[Path]:
    """Create a realistic, isolated Git repo for ``act`` runs.

    We copy the full working tree (minus ephemeral/build artifacts) so that
    workflows referencing files like ``pyproject.toml`` or ``Makefile`` can run
    as they would against the real repository.
    """
    repo_path = tmp_path / "repo"

    # Mirror the working tree with sensible ignores to keep the repo lean
    source_root = Path.cwd()
    ignore = shutil.ignore_patterns(
        ".git",
        "**/__pycache__",
        "*.pyc",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".coverage",
        "coverage_html_report",
        "dist",
        "build",
        ".tox",
        ".venv",
        "venv",
        "env",
        ".env",
        ".DS_Store",
        # Large, non-essential for workflow execution speed
        "site",  # MkDocs build output
        "test_files",  # Media fixtures
        "test_data",  # e.g., examples/test_data
        "notebooks",
        "*.ipynb",
    )
    shutil.copytree(source_root, repo_path, dirs_exist_ok=True, ignore=ignore)

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            command,
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )

    # Ensure a consistent default branch across environments
    run(["git", "init", "--initial-branch=main"])
    run(["git", "remote", "add", "origin", "https://github.com/user/repo.git"])
    run(["git", "config", "user.name", "Test User"])
    run(["git", "config", "user.email", "test@example.com"])
    run(["git", "add", "."])
    run(["git", "commit", "-m", "initial commit"])
    run(["git", "tag", "v0.1.0"])  # Tag the first commit

    yield repo_path


@pytest.fixture
def release_workflow():
    with Path(".github/workflows/release.yml").open() as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert "workflow_dispatch" in data.get("on", {})
    return data


@pytest.fixture
def reusable_checks_workflow():
    with Path(".github/workflows/reusable-checks.yml").open() as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert "workflow_call" in data.get("on", {})
    return data


@pytest.fixture
def ci_workflow():
    with Path(".github/workflows/ci.yml").open() as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    return data


@pytest.fixture
def semantic_release_config():
    with Path("pyproject.toml").open("rb") as f:
        config = tomllib.load(f)
    assert isinstance(config, dict)
    sr_config = config["tool"]["semantic_release"]
    assert isinstance(sr_config, dict)
    return sr_config


@pytest.fixture
def project_config():
    with Path("pyproject.toml").open("rb") as f:
        config = tomllib.load(f)
    assert isinstance(config, dict)
    project_config = config["project"]
    assert isinstance(project_config, dict)
    return project_config


@pytest.fixture
def reusable_checks_inputs():
    workflow_path = Path(".github/workflows/reusable-checks.yml")
    with workflow_path.open() as f:
        workflow = yaml.safe_load(f)
    required_inputs = set()
    inputs = workflow["on"]["workflow_call"]["inputs"]
    for name, input_def in inputs.items():
        if input_def.get("required", False):
            required_inputs.add(name)
    return required_inputs


@pytest.fixture
def all_workflows():
    workflow_dir = Path(".github/workflows")
    workflows = {}
    for workflow_file in workflow_dir.glob("*.yml"):
        with workflow_file.open() as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        workflows[workflow_file.name] = data
    return workflows


@pytest.fixture
def pyproject_python_version() -> str:
    with Path("pyproject.toml").open("rb") as f:
        config = tomllib.load(f)
    raw = str(config.get("project", {}).get("requires-python", ""))
    # Normalize ">=3.13" to "3.13" to match workflow pin style
    return raw.lstrip(">=").strip()


@pytest.fixture
def makefile_targets() -> set[str]:
    """Parse Makefile and return a set of declared targets."""
    mk = Path("Makefile").read_text(encoding="utf-8")
    targets: set[str] = set()
    for line in mk.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+):", line)
        if m and m.group(1) != ".PHONY":
            targets.add(m.group(1))
    return targets


@pytest.fixture
def github_token_for_act() -> str:
    """Provide a GitHub token for act or skip if unavailable.

    A real token significantly improves reliability when ``act`` pulls actions
    and performs authenticated Git operations. If ``GITHUB_TOKEN_FOR_ACT`` is
    not set in the environment, we skip tests that require it rather than
    failing with environment-specific auth errors.
    """
    token = os.getenv("GITHUB_TOKEN_FOR_ACT")
    if not token:
        pytest.skip(
            "Set GITHUB_TOKEN_FOR_ACT to run workflow execution tests requiring auth"
        )
    return token


@pytest.fixture
def act_helper(act_executable: str, initialized_git_repo: Path) -> ActTestHelper:
    """High-level helper to run workflows via act in the temp git repo."""
    git = GitHelper("git", initialized_git_repo)
    return ActTestHelper(act_executable, git)
