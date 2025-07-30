"""
Global test configuration with support for different test types.
"""

from collections.abc import Generator
from datetime import datetime
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tomllib
import typing
from unittest.mock import MagicMock, PropertyMock, patch

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
import pytest
import yaml

from gemini_batch import BatchProcessor, GeminiClient
from tests.helpers import ActTestHelper, GitHelper, MockCommit


# --- Logging Fixtures ---
@pytest.fixture(scope="session", autouse=True)
def quiet_noisy_libraries():
    """Sets the log level for noisy external libraries to WARNING."""
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


# --- Test Environment Markers ---
def pytest_configure(config):
    """Configure custom markers for test organization."""
    markers = [
        "unit: Fast, isolated unit tests",
        "integration: Component integration tests with mocked APIs",
        "api: Real API integration tests (requires API key)",
        "characterization: Golden master tests to detect behavior changes.",
        "slow: Tests that take >1 second",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Automatically skip API tests when API key is unavailable."""
    if not (os.getenv("GEMINI_API_KEY") and os.getenv("ENABLE_API_TESTS")):
        skip_api = pytest.mark.skip(
            reason="API tests require GEMINI_API_KEY and ENABLE_API_TESTS=1",
        )
        for item in items:
            if "api" in item.keywords:
                item.add_marker(skip_api)


# --- Core Fixtures ---


@pytest.fixture
def mock_api_key():
    """Provide a consistent, fake API key for tests."""
    return "test_api_key_12345_67890_abcdef_ghijkl"


@pytest.fixture
def mock_env(mock_api_key, monkeypatch):
    """
    Mocks essential environment variables to ensure tests run in a
    consistent, isolated environment.
    """
    monkeypatch.setenv("GEMINI_API_KEY", mock_api_key)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("GEMINI_ENABLE_CACHING", "False")


@pytest.fixture
def mock_gemini_client(mock_env):  # noqa: ARG001
    """
    Provides a MagicMock of the GeminiClient.

    This is a powerful fixture that prevents real API calls. The mock's methods,
    like `generate_content`, can be configured within individual tests to return
    specific, predictable responses, making tests deterministic and fast.
    """
    # We create a mock instance that has the same interface as the real GeminiClient.
    mock_client = MagicMock(spec=GeminiClient)

    # We mock the `generate_content` method, as this is the primary method
    # called by BatchProcessor.
    mock_client.generate_content.return_value = {
        "text": '["Default mock answer"]',
        "usage": {
            "prompt_tokens": 10,
            "candidates_token_count": 5,
            "total_tokens": 15,
        },
    }
    return mock_client


@pytest.fixture
def batch_processor(mock_gemini_client):
    """
    Provides a BatchProcessor instance that is pre-configured with a
    mocked GeminiClient. This is the standard way to test BatchProcessor
    without hitting the actual API.
    """
    # We inject the mock client directly into the processor.
    # This is a key principle of Dependency Injection for testability.
    return BatchProcessor(_client=mock_gemini_client)


# --- Advanced Fixtures for Client Behavior Testing ---


@pytest.fixture
def mocked_internal_genai_client():
    """
    Mocks the internal `google.genai.Client` that GeminiClient uses.

    This allows us to test the logic of our GeminiClient by inspecting the
    low-level API calls it attempts to make, without any real network activity.
    """
    # We patch the class that our client will instantiate.
    with patch("gemini_batch.gemini_client.genai.Client") as mock_genai:
        # Create an instance of the mock to be used by our client
        mock_instance = mock_genai.return_value

        # Mock the nested structure for creating and using caches
        mock_instance.caches.create.return_value = MagicMock(
            name="caches.create_return",
        )
        type(mock_instance.caches.create.return_value).name = PropertyMock(
            return_value="cachedContents/mock-cache-123",
        )

        # Mock the token counter to avoid real API calls during planning
        mock_instance.models.count_tokens.return_value = MagicMock(total_tokens=5000)

        yield mock_instance


@pytest.fixture
def caching_gemini_client(mock_env, mocked_internal_genai_client):  # noqa: ARG001
    """
    Provides a real GeminiClient instance configured for caching, but with
    its internal API calls mocked.
    """
    # We instantiate our actual GeminiClient. Because the `genai.Client` is
    # patched, our client will receive the `mocked_internal_genai_client`
    # when it tries to initialize its internal client.
    # We explicitly enable caching for this test client.
    client = GeminiClient(enable_caching=True)
    return client  # noqa: RET504


@pytest.fixture
def mock_httpx_client():
    """
    Mocks the httpx.Client to prevent real network requests for URL processing.
    """
    # We patch the client where it's used in the extractors module.
    with patch("gemini_batch.files.extractors.httpx.Client") as mock_client_class:
        mock_instance = mock_client_class.return_value.__enter__.return_value
        yield mock_instance


@pytest.fixture
def mock_get_mime_type():
    """
    Mocks the get_mime_type utility function to prevent python-magic
    from bypassing pyfakefs, making MIME type detection deterministic in tests.
    """
    # We patch the function in the `utils` module where it is defined.
    with patch("gemini_batch.files.utils.get_mime_type") as mock_func:
        # Define a simple side effect to simulate MIME type detection based on file extension.
        def side_effect(file_path, use_magic=True):  # noqa: ARG001, FBT002
            if str(file_path).endswith(".png"):
                return "image/png"
            if str(file_path).endswith(".txt") or str(file_path).endswith(".md"):
                return "text/plain"
            # Provide a sensible default for any other file types in tests.
            return "application/octet-stream"

        mock_func.side_effect = side_effect
        yield mock_func


# --- Helper Fixtures ---
@pytest.fixture
def fs(fs):
    """
    A fixture for pyfakefs that automatically enables OS-specific path separators.
    This makes filesystem tests more robust across different operating systems.
    """
    fs.os = os
    return fs


# --- Integration Fixtures ---
@pytest.fixture
def initialized_git_repo(tmp_path: Path) -> Generator[Path]:
    """
    Creates a temporary, isolated Git repository configured for semantic-release.

    This fixture provides a clean starting point for integration tests
    that need to run semantic-release commands. It yields the path to the
    repo, which is automatically cleaned up by pytest.
    """
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    # --- Create a minimal configuration for semantic-release ---
    config_content = """
    [tool.semantic_release]
    version_source = "file"
    version_variable = "VERSION"
    changelog_file = "CHANGELOG.md"
    """
    (repo_path / "pyproject.toml").write_text(config_content)
    (repo_path / "VERSION").write_text("0.1.0")

    # Helper to run commands within the repo
    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603  # The command is a hardcoded list of strings, not user input.
            command,
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )

    # Initialize and configure the Git repository
    run(["git", "init"])
    run(["git", "remote", "add", "origin", "https://github.com/user/repo.git"])
    run(["git", "config", "user.name", "Test User"])
    run(["git", "config", "user.email", "test@example.com"])
    run(["git", "add", "."])
    run(["git", "commit", "-m", "initial commit"])
    run(["git", "tag", "v0.1.0"])  # Tag the first commit

    yield repo_path


# --- Pydantic Schema for Structured Tests ---
class SimpleSummary(BaseModel):
    """A simple Pydantic schema for structured output tests."""

    summary: str
    key_points: list[str]


# --- Changelog Fixtures ---
@pytest.fixture(scope="module")
def jinja_env():
    """Provides a configured Jinja2 environment for template tests."""
    template_path = Path("templates")
    if not template_path.exists():
        pytest.skip("Changelog template directory not found.")

    # Create a Jinja environment similar to semantic-release's environment,
    # matching settings in pyproject.toml
    env = Environment(  # noqa: S701
        loader=FileSystemLoader(searchpath=str(template_path)),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    env.filters["commit_hash_url"] = (
        lambda x: f"https://github.com/USER/REPO/commit/{x}"
    )
    env.filters["issue_url"] = (
        lambda x: f"https://github.com/USER/REPO/issues/{x.lstrip('#')}"
    )
    env.filters["compare_url"] = (
        lambda x, y: f"https://github.com/USER/REPO/compare/{x}...{y}"
    )

    return env


@pytest.fixture
def macros(jinja_env):
    """Fixture to load the macros from the template file for direct testing."""
    # This loads the macros and makes them available as attributes on the 'module' object
    return jinja_env.get_template(".macros.j2").module


@pytest.fixture
def mock_changelog_context():
    """
    Provides a mock context dictionary using a robust Pydantic model.
    This emulates the data passed by python-semantic-release.
    """
    # Create a structure mimicking semantic-release's history using the robust model
    # See: https://python-semantic-release.readthedocs.io/en/latest/concepts/changelog_templates.html
    return {
        "history": {
            "unreleased": {
                "features": [
                    MockCommit(
                        type="feat",
                        scope="api",
                        descriptions=["add exciting new endpoint"],
                        short_hash="f34t001",
                        hexsha="f34t001" * 5,
                        breaking_descriptions=[],
                    ),
                    MockCommit(
                        type="feat",
                        scope="client",
                        descriptions=["link to external issues"],
                        short_hash="f34t002",
                        hexsha="f34t002" * 5,
                        linked_issues=["#123", "#456"],
                    ),
                ],
                "bug fixes": [
                    MockCommit(
                        type="fix",
                        scope="client",
                        descriptions=["resolve critical connection bug"],
                        short_hash="f18x001",
                        hexsha="f18x001" * 5,
                    ),
                    MockCommit(
                        type="fix",
                        scope="",
                        descriptions=["correct a minor typo in the README"],
                        short_hash="f18x002",
                        hexsha="f18x002" * 5,
                    ),
                ],
                "refactoring": [
                    MockCommit(
                        type="refactor",
                        scope="core",
                        descriptions=["simplify internal logic"],
                        short_hash="r3f4c70",
                        hexsha="r3f4c70" * 5,
                    ),
                    MockCommit(
                        type="refactor",
                        scope="auth",
                        descriptions=["rework authentication flow"],
                        short_hash="r3f4c71",
                        hexsha="r3f4c71" * 5,
                        breaking_descriptions=[
                            "A breaking change related to rework authentication flow."
                        ],
                    ),
                ],
            },
            "released": {
                "1.0.0": {
                    "version": "1.0.0",
                    "tagged_date": datetime(2024, 7, 1).date(),
                    "elements": {
                        "features": [
                            MockCommit(
                                type="feat",
                                scope="core",
                                descriptions=["implement initial logic"],
                                short_hash="b4dfeat",
                                hexsha="b4dfeat" * 5,
                                breaking_descriptions=[
                                    "A breaking change related to implement initial logic."
                                ],
                            ),
                        ],
                        "documentation": [
                            MockCommit(
                                type="docs",
                                scope="",
                                descriptions=["add README and contributing guide"],
                                short_hash="d0c5001",
                                hexsha="d0c5001" * 5,
                            ),
                        ],
                        "build system": [
                            MockCommit(
                                type="build",
                                scope="",
                                descriptions=["configure pyproject.toml"],
                                short_hash="b51ld01",
                                hexsha="b51ld01" * 5,
                            ),
                        ],
                    },
                }
            },
        }
    }


@pytest.fixture
def mock_changelog_context_no_breaking():
    """Context with commits but no breaking changes."""
    return {
        "history": {
            "unreleased": {
                "bug fixes": [
                    MockCommit(
                        type="fix",
                        scope="client",
                        descriptions=["standardize usage metadata key"],
                        short_hash="dd7b3e8",
                        hexsha="dd7b3e8" * 5,
                        breaking_descriptions=[],  # ← No breaking changes
                    ),
                ],
                "features": [],
                "refactoring": [],
            },
            "released": {
                "1.0.0": {
                    "version": "1.0.0",
                    "tagged_date": datetime(2024, 7, 1).date(),
                    "elements": {
                        "features": [
                            MockCommit(
                                type="feat",
                                scope="core",
                                descriptions=["implement initial logic"],
                                short_hash="b4dfeat",
                                hexsha="b4dfeat" * 5,
                                breaking_descriptions=[
                                    "A breaking change related to implement initial logic."
                                ],
                            ),
                        ],
                        "documentation": [
                            MockCommit(
                                type="docs",
                                scope="",
                                descriptions=["add README and contributing guide"],
                                short_hash="d0c5001",
                                hexsha="d0c5001" * 5,
                            ),
                        ],
                        "build system": [
                            MockCommit(
                                type="build",
                                scope="",
                                descriptions=["configure pyproject.toml"],
                                short_hash="b51ld01",
                                hexsha="b51ld01" * 5,
                            ),
                        ],
                    },
                }
            },
        }
    }


@pytest.fixture
def mock_changelog_context_initial_release():
    """
    Context for testing the very first release when there is nothing to compare against.
    This has only one released version and no 'unreleased' section.
    """
    return {
        "history": {
            "unreleased": {},  # No unreleased commits
            "released": {
                "1.0.0": {
                    "version": "1.0.0",
                    "tagged_date": datetime(2024, 7, 1).date(),
                    "elements": {
                        "features": [
                            MockCommit(
                                type="feat",
                                scope="core",
                                descriptions=["implement initial logic"],
                                short_hash="b4dfeat",
                                hexsha="b4dfeat" * 5,
                                breaking_descriptions=[],
                            ),
                        ],
                        "documentation": [
                            MockCommit(
                                type="docs",
                                scope="",
                                descriptions=["add README and contributing guide"],
                                short_hash="d0c5001",
                                hexsha="d0c5001" * 5,
                            ),
                        ],
                    },
                }
            },
        }
    }


@pytest.fixture
def mock_changelog_context_multiple_releases():
    """
    Context for testing multiple releases with comparison links.
    This has at least two released versions: 1.1.0 and 1.0.0.
    """
    return {
        "history": {
            "unreleased": {
                "features": [
                    MockCommit(
                        type="feat",
                        scope="api",
                        descriptions=["add exciting new endpoint"],
                        short_hash="f34t001",
                        hexsha="f34t001" * 5,
                        breaking_descriptions=[],
                    ),
                ],
                "bug fixes": [],
                "refactoring": [],
                "performance improvements": [],
                "reverts": [],
                "documentation": [],
                "build system": [],
            },
            "released": {
                "1.1.0": {
                    "version": "1.1.0",
                    "tagged_date": datetime(2024, 8, 1).date(),
                    "elements": {
                        "features": [
                            MockCommit(
                                type="feat",
                                scope="client",
                                descriptions=["add client improvements"],
                                short_hash="f34t002",
                                hexsha="f34t002" * 5,
                                breaking_descriptions=[],
                            ),
                        ],
                        "bug fixes": [
                            MockCommit(
                                type="fix",
                                scope="api",
                                descriptions=["fix API response format"],
                                short_hash="f18x003",
                                hexsha="f18x003" * 5,
                            ),
                        ],
                    },
                },
                "1.0.0": {
                    "version": "1.0.0",
                    "tagged_date": datetime(2024, 7, 1).date(),
                    "elements": {
                        "features": [
                            MockCommit(
                                type="feat",
                                scope="core",
                                descriptions=["implement initial logic"],
                                short_hash="b4dfeat",
                                hexsha="b4dfeat" * 5,
                                breaking_descriptions=[],
                            ),
                        ],
                        "documentation": [
                            MockCommit(
                                type="docs",
                                scope="",
                                descriptions=["add README and contributing guide"],
                                short_hash="d0c5001",
                                hexsha="d0c5001" * 5,
                            ),
                        ],
                    },
                },
            },
        }
    }


# --- Workflow Fixtures ---
@pytest.fixture(scope="session")
def act_executable() -> str:
    """
    Finds the full path to the 'act' executable and skips the test if not found.
    """
    path = shutil.which("act")
    if not path:
        pytest.skip("The 'act' executable was not found in the system's PATH.")
    return path


@pytest.fixture
def release_workflow():
    """Load the release workflow."""
    with Path(".github/workflows/release.yml").open() as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), "Release workflow must be a dictionary"
    assert "workflow_dispatch" in data.get("on", {}), "Must be a release workflow"
    return data


@pytest.fixture
def reusable_checks_workflow():
    """Load the reusable checks workflow with proper typing."""
    with Path(".github/workflows/reusable-checks.yml").open() as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), "Reusable checks workflow must be a dictionary"
    # Runtime validation that this is actually a reusable workflow
    assert "workflow_call" in data.get("on", {}), "Must be a reusable workflow"

    return data


@pytest.fixture
def ci_workflow():
    """Load the CI workflow with proper typing."""
    with Path(".github/workflows/ci.yml").open() as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), "CI workflow must be a dictionary"
    return data


@pytest.fixture
def semantic_release_config():
    """Load semantic-release configuration."""
    with Path("pyproject.toml").open("rb") as f:
        config = tomllib.load(f)

    assert isinstance(config, dict), "Configuration must be a dictionary"
    sr_config = config["tool"]["semantic_release"]
    assert isinstance(sr_config, dict), (
        "Semantic release configuration must be a dictionary"
    )
    return sr_config


@pytest.fixture
def project_config():
    """Load project configuration with proper typing."""
    with Path("pyproject.toml").open("rb") as f:
        config = tomllib.load(f)

    assert isinstance(config, dict), "Project configuration must be a dictionary"
    project_config = config["project"]
    assert isinstance(project_config, dict), "Project section must be a dictionary"

    return project_config


@pytest.fixture
def reusable_checks_inputs():
    """Load required inputs from reusable-checks.yml."""
    workflow_path = Path(".github/workflows/reusable-checks.yml")
    with workflow_path.open() as f:
        workflow = yaml.safe_load(f)

    required_inputs = set()
    inputs = workflow["on"]["workflow_call"]["inputs"]
    for input_name, input_def in inputs.items():
        if input_def.get("required", False):
            required_inputs.add(input_name)

    return required_inputs


@pytest.fixture
def all_workflows():
    """Load all workflow files with proper typing."""
    workflow_dir = Path(".github/workflows")
    workflows = {}

    for workflow_file in workflow_dir.glob("*.yml"):
        with workflow_file.open() as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"{workflow_file.name} must be a dictionary"
        workflows[workflow_file.name] = data

    return workflows


@pytest.fixture
def pyproject_python_version() -> str:
    """Get the Python version from pyproject.toml."""
    with Path("pyproject.toml").open("rb") as f:
        config = tomllib.load(f)

    # Extract minimum Python version from requires-python
    requires_python = config["project"]["requires-python"]
    assert isinstance(requires_python, str), "Requires Python must be a string"
    # Parse ">=3.13" -> "3.13"
    return requires_python.replace(">=", "").strip()


@pytest.fixture
def makefile_targets() -> set[str]:
    """Extract available targets from Makefile."""
    if not Path("Makefile").exists():
        pytest.skip("Makefile not found")

    with Path("Makefile").open() as f:
        content = f.read()

    # Simple extraction of .PHONY targets and explicit targets
    targets = set()
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith(".PHONY:"):
            phony_targets = line.replace(".PHONY:", "").strip().split()
            targets.update(phony_targets)
        elif ":" in line and not line.startswith("\t") and not line.startswith("#"):
            target = line.split(":")[0].strip()
            if target and not target.startswith("."):
                targets.add(target)

    return targets


@pytest.fixture(scope="session")
def git_executable() -> str:
    """
    Finds the git executable and skips tests if not found.
    """
    path = shutil.which("git")
    if not path:
        pytest.skip("The 'git' executable was not found in the system's PATH.")
    return path


@pytest.fixture
def git_helper(
    git_executable: str,
    e2e_git_repo: tuple[
        Path, typing.Callable[[list[str]], subprocess.CompletedProcess[str]]
    ],
) -> "GitHelper":
    """Provides a GitHelper for the test repository."""

    repo_path, _ = e2e_git_repo
    return GitHelper(git_executable, repo_path)


@pytest.fixture
def act_helper(act_executable: str, git_helper: "GitHelper") -> "ActTestHelper":
    """Provides a high-level ActTestHelper."""

    return ActTestHelper(act_executable, git_helper)


@pytest.fixture
def github_token_for_act() -> str:
    """Get GitHub token for act, skipping test if not available."""
    token = os.getenv("GITHUB_TOKEN_FOR_ACT")
    if not token:
        pytest.skip("Set GITHUB_TOKEN_FOR_ACT environment variable to run this test")
    return token


@pytest.fixture
def e2e_git_repo(
    tmp_path: Path,
    git_executable: str,
) -> Generator[tuple[Path, "GitHelper"]]:
    """
    Creates a temporary, isolated Git repository with a full copy of the
    project's essential files.

    This provides a high-fidelity environment for end-to-end workflow tests
    using 'act'. It yields a GitHelper instance.
    """
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    # --- Copy all essential project files to the isolated environment ---
    essential_files = ["pyproject.toml", "Makefile", "README.md", "LICENSE"]
    essential_dirs = [".github", "src", "tests", "templates"]

    for file in essential_files:
        if Path(file).exists():
            shutil.copy(file, repo_path)

    for directory in essential_dirs:
        if Path(directory).is_dir():
            shutil.copytree(directory, repo_path / directory)

    # --- Create GitHelper and use it for setup ---
    git_helper = GitHelper(git_executable, repo_path)

    # --- Initialize and configure the Git repository ---
    git_helper.run(["init", "--initial-branch=main"])
    git_helper.run(["remote", "add", "origin", "https://github.com/user/repo.git"])
    git_helper.run(["config", "user.name", "Test User"])
    git_helper.run(["config", "user.email", "test@example.com"])
    git_helper.run(["add", "."])
    git_helper.run(["commit", "-m", "chore: initial commit of project files"])
    git_helper.run(["tag", "v0.1.0"])

    yield repo_path, git_helper
