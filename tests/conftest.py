"""
Global test configuration with support for different test types.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
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

from tests.helpers import ActTestHelper, GitHelper, MockCommit

# Note: Old BatchProcessor and GeminiClient imports removed during transition
# The test adapter provides the interface compatibility needed for existing tests


# --- Environment Isolation (Autouse) ---
@pytest.fixture(autouse=True)
def block_dotenv(request, monkeypatch):
    """Prevent python-dotenv from loading project .env files during tests.

    Architecture Rubric: remove hidden state and action-at-a-distance. Tests
    should only see environment that they explicitly set.

    Opt-in escape hatch: mark a test with @pytest.mark.allow_dotenv
    to permit .env loading for that specific test.
    """
    if request.node.get_closest_marker("allow_dotenv"):
        return
    # If dotenv is installed, replace load_dotenv with a no-op for this test.
    with suppress(Exception):
        monkeypatch.setattr(
            "dotenv.load_dotenv", lambda *_args, **_kwargs: False, raising=False
        )


@pytest.fixture(autouse=True)
def isolate_gemini_env(request, monkeypatch):
    """Ensure a clean GEMINI_* environment for each test.

    - Removes all GEMINI_* variables and debug toggles before each test
    - Leaves non-GEMINI_* variables intact for stability

    Escape hatches:
      - @pytest.mark.allow_env_pollution: keep current env unchanged
      - tests marked with @pytest.mark.api automatically bypass isolation
        so real environment can be used when explicitly running API tests.
    """
    if request.node.get_closest_marker("allow_env_pollution") or (
        "api" in request.node.keywords
    ):
        return

    for key in list(os.environ.keys()):
        if key.startswith("GEMINI_"):
            monkeypatch.delenv(key, raising=False)
    # Avoid DEBUG toggles affecting telemetry/config debug paths
    monkeypatch.delenv("DEBUG", raising=False)
    monkeypatch.delenv("GEMINI_BATCH_DEBUG_CONFIG", raising=False)


@pytest.fixture(autouse=True)
def neutral_home_config(request, monkeypatch, tmp_path):
    """Point home-config path to an isolated temp file by default.

    Prevents reading a developer's real ~/.config/gemini_batch.toml during tests.

    Escape hatch: mark test with @pytest.mark.allow_real_home_config to use
    the real path.
    """
    if request.node.get_closest_marker("allow_real_home_config"):
        return

    fake_home_dir = tmp_path / "home_config_isolated"
    fake_home_dir.mkdir(parents=True, exist_ok=True)
    fake_home_file = fake_home_dir / "gemini_batch.toml"
    # Prefer environment override to avoid monkeypatching internals
    monkeypatch.setenv("GEMINI_BATCH_CONFIG_HOME", str(fake_home_file))


@pytest.fixture
def clean_env_patch():
    """Helper to apply a clean env baseline plus overrides.

    Usage:
        with clean_env_patch({"GEMINI_BATCH_MODEL": "env-model"}):
            ...
    """

    def _apply(extra: dict[str, str] | None = None) -> Generator[None]:
        base = {k: v for k, v in os.environ.items() if not k.startswith("GEMINI_")}
        if extra:
            base.update(extra)
        with patch.dict(os.environ, base, clear=True):
            yield

    return _apply


@pytest.fixture
def isolated_config_sources(tmp_path):
    """Completely isolate configuration sources for testing.

    This fixture ensures that:
    1. No environment variables affect config resolution
    2. No real home config is loaded
    3. No real pyproject.toml is loaded
    4. Only explicitly provided configs are used

    Returns a helper function to set up specific config sources.
    """

    @contextmanager
    def _setup(
        *,
        pyproject_content: str = "",
        home_content: str = "",
        env_vars: dict[str, str] | None = None,
    ) -> Generator[None]:
        """Set up isolated config sources with specific content.

        Args:
            pyproject_content: TOML content for pyproject file
            home_content: TOML content for home config file
            env_vars: Environment variables to set (GEMINI_ prefix added automatically)
        """
        # Clean environment
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("GEMINI_")}
        if env_vars:
            for key, value in env_vars.items():
                if not key.startswith("GEMINI_"):
                    key = f"GEMINI_BATCH_{key.upper()}"
                clean_env[key] = value

        # Set up file paths
        project_dir = tmp_path / "project"
        project_dir.mkdir(exist_ok=True)
        pyproject_path = project_dir / "pyproject.toml"

        home_dir = tmp_path / "home"
        home_dir.mkdir(exist_ok=True)
        home_config_path = home_dir / "gemini_batch.toml"

        # Write config files if content provided
        if pyproject_content:
            pyproject_path.write_text(pyproject_content)
        if home_content:
            home_config_path.write_text(home_content)

        # Prefer environment overrides for paths to avoid monkeypatching internals
        clean_env["GEMINI_BATCH_PYPROJECT_PATH"] = str(pyproject_path)
        clean_env["GEMINI_BATCH_CONFIG_HOME"] = str(home_config_path)

        # Apply environment and yield control
        with patch.dict(os.environ, clean_env, clear=True):
            yield

    return _setup


@pytest.fixture
def temp_toml_file():
    """Create temporary TOML files for testing.

    Returns a context manager that creates a temp file with given content.
    """
    from contextlib import contextmanager
    import tempfile

    @contextmanager
    def _create(content: str) -> Generator[Path]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(content)
            f.flush()
            temp_path = Path(f.name)
        try:
            yield temp_path
        finally:
            if temp_path.exists():
                temp_path.unlink()

    return _create


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
    if not (
        (os.getenv("GEMINI_BATCH_API_KEY") or os.getenv("GEMINI_API_KEY"))
        and os.getenv("ENABLE_API_TESTS")
    ):
        skip_api = pytest.mark.skip(
            reason="API tests require GEMINI_BATCH_API_KEY or GEMINI_API_KEY and ENABLE_API_TESTS=1",
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
    monkeypatch.setenv("GEMINI_BATCH_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("GEMINI_BATCH_ENABLE_CACHING", "False")


@pytest.fixture
def mock_gemini_client(mock_env):  # noqa: ARG001
    """
    Provides a MagicMock for the old GeminiClient interface.

    This fixture is kept for backward compatibility with existing tests,
    but the actual mocking is now handled by the new architecture's test adapter.
    """
    # Create a generic mock that can be configured by tests
    mock_client = MagicMock()

    # Set up default return values for the old interface
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
def batch_processor(mock_gemini_client):  # noqa: ARG001
    """
    Provides a BatchProcessor-like object for testing using the new executor.

    This keeps characterization tests unchanged while removing the tests/adapters.py file.
    """
    from typing import Any

    from gemini_batch.config import resolve_config
    from gemini_batch.core.types import InitialCommand, ResultEnvelope
    from gemini_batch.executor import GeminiExecutor

    class _TestAdapterBatchProcessor:
        def __init__(self, **config_overrides: Any):
            programmatic: dict[str, Any] = {
                "api_key": config_overrides.get("api_key", "mock_api_key_for_tests"),
                "model": config_overrides.get("model", "gemini-2.0-flash"),
                "enable_caching": config_overrides.get("enable_caching", False),
                "use_real_api": config_overrides.get("use_real_api", False),
            }
            if "tier" in config_overrides:
                programmatic["tier"] = config_overrides["tier"]
            if "ttl_seconds" in config_overrides:
                programmatic["ttl_seconds"] = config_overrides["ttl_seconds"]

            self.config = resolve_config(overrides=programmatic)
            self.executor = GeminiExecutor(config=self.config)

        def process_questions(
            self,
            content: Any,
            questions: list[str],
            _compare_methods: bool = False,  # noqa: FBT001, FBT002
            _response_schema: Any | None = None,
            _return_usage: bool = False,  # noqa: FBT001, FBT002
            **_kwargs: Any,
        ) -> ResultEnvelope:
            sources = tuple(content) if isinstance(content, list) else (content,)
            prompts = tuple(questions)
            command = InitialCommand(
                sources=sources, prompts=prompts, config=self.config
            )

            import asyncio

            try:
                return asyncio.run(self.executor.execute(command))
            except Exception as e:
                return {
                    "status": "error",
                    "answers": [],
                    "extraction_method": "error",
                    "confidence": 0.0,
                    "metrics": {"error": str(e)},
                    "usage": {},
                }

        def process_questions_multi_source(
            self,
            sources: list[Any],
            questions: list[str],
            response_schema: Any | None = None,
            **kwargs: Any,
        ) -> ResultEnvelope:
            flat_sources: list[Any] = []
            for src in sources:
                if isinstance(src, list):
                    flat_sources.extend(src)
                else:
                    flat_sources.append(src)
            return self.process_questions(
                content=flat_sources,
                questions=questions,
                response_schema=response_schema,
                **kwargs,
            )

    return _TestAdapterBatchProcessor(api_key="mock_api_key_for_tests")


# --- Characterization Executor Fixture (new arch) ---


@pytest.fixture
def char_executor(mock_gemini_client):
    """Provide a GeminiExecutor wired with a controllable test adapter.

    - Uses the new pipeline directly.
    - Adapter returns values from the legacy `mock_gemini_client` when set
      via its `generate_content.side_effect` or `return_value`.
    - Implements a trivial cache capability so tests can observe cache
      creation and application when desired.
    """
    from typing import Any, cast

    from gemini_batch.config import resolve_config
    from gemini_batch.executor import GeminiExecutor
    from gemini_batch.pipeline.adapters.base import (
        CachingCapability,
        ExecutionHintsAware,
        GenerationAdapter,
    )
    from gemini_batch.pipeline.api_handler import APIHandler
    from gemini_batch.pipeline.cache_stage import CacheStage
    from gemini_batch.pipeline.execution_state import ExecutionHints
    from gemini_batch.pipeline.rate_limit_handler import RateLimitHandler
    from gemini_batch.pipeline.result_builder import ResultBuilder
    from gemini_batch.pipeline.source_handler import SourceHandler

    class _Adapter(GenerationAdapter, CachingCapability, ExecutionHintsAware):
        def __init__(self) -> None:
            self._hints: ExecutionHints | None = None
            self.interaction_log: list[dict[str, object]] | None = None
            self.queue: list[dict[str, Any]] = []

        def apply_hints(self, hints: Any) -> None:
            if isinstance(hints, ExecutionHints):
                self._hints = hints

        async def create_cache(
            self,
            *,
            model_name: str,  # noqa: ARG002
            content_parts: tuple[Any, ...],  # noqa: ARG002
            system_instruction: str | None,  # noqa: ARG002
            ttl_seconds: int | None,  # noqa: ARG002
        ) -> str:
            name = "cachedContents/mock-cache-123"
            if self.interaction_log is not None:
                self.interaction_log.append({"method": "caches.create"})
            return name

        async def generate(
            self,
            *,
            model_name: str,  # noqa: ARG002
            api_parts: tuple[Any, ...],
            api_config: dict[str, object],
        ) -> dict[str, Any]:
            # Prefer explicit queued responses set by tests
            if self.queue:
                return self.queue.pop(0)

            # Record generate call and any applied cache
            if self.interaction_log is not None:
                cached_value = cast("str | None", api_config.get("cached_content"))
                entry: dict[str, object] = {"method": "generate_content"}
                if cached_value is not None:
                    entry["cached_content"] = cached_value
                self.interaction_log.append(entry)

            # Bridge to legacy mock if configured by tests
            if hasattr(mock_gemini_client, "generate_content"):
                fn = mock_gemini_client.generate_content
                se = getattr(fn, "side_effect", None)
                if callable(se) or isinstance(se, list):
                    if isinstance(se, list):
                        # Pop sequential results if list provided
                        return cast("dict[str, Any]", se.pop(0))
                    return cast("dict[str, Any]", fn.side_effect())
                rv = getattr(fn, "return_value", None)
                if isinstance(rv, dict):
                    return cast("dict[str, Any]", rv)

            # Default minimal response
            text = ""
            try:
                part0 = next(iter(api_parts))
                text = getattr(part0, "text", str(part0))
            except StopIteration:
                text = ""
            return {"text": text, "usage": {"total_token_count": 0}}

    adapter = _Adapter()

    def make_executor(
        *, interaction_log: list[dict[str, object]] | None = None
    ) -> GeminiExecutor:
        from gemini_batch.pipeline.planner import ExecutionPlanner

        cfg = resolve_config(
            overrides={
                "api_key": "mock_api_key_for_tests",
                "model": "gemini-2.0-flash",
                "enable_caching": True,
                "use_real_api": False,
            }
        )
        adapter.interaction_log = interaction_log
        from gemini_batch.pipeline.registries import CacheRegistry

        cache_reg = CacheRegistry()
        pipeline: list[Any] = [
            SourceHandler(),
            ExecutionPlanner(),
            RateLimitHandler(),
            CacheStage(
                registries={"cache": cache_reg}, adapter_factory=lambda _k: adapter
            ),
            APIHandler(adapter=adapter, registries={"cache": cache_reg}),
            ResultBuilder(),
        ]
        return GeminiExecutor(cfg, pipeline_handlers=pipeline)

    class _Exec:
        def __init__(self, factory: Callable[..., GeminiExecutor]):
            self._factory = factory
            self.adapter = adapter

        def build(
            self, *, interaction_log: list[dict[str, object]] | None = None
        ) -> GeminiExecutor:
            # Forward provided interaction_log into the executor factory so
            # the adapter can record interactions for characterization tests.
            return self._factory(interaction_log=interaction_log)

    return _Exec(make_executor)


# --- Advanced Fixtures for Client Behavior Testing ---


@pytest.fixture
def mocked_internal_genai_client():
    """
    Mocks the internal `google.genai.Client` that GeminiClient uses.

    This allows us to test the logic of our GeminiClient by inspecting the
    low-level API calls it attempts to make, without any real network activity.
    """
    # We patch the class that our client will instantiate.
    with patch("google.genai.Client") as mock_genai:
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
    Provides a mock client configured for caching testing.

    This fixture is kept for backward compatibility but now returns a mock
    since the new architecture handles caching differently.
    """
    # Return a mock client for backward compatibility
    mock_client = MagicMock()
    mock_client.enable_caching = True
    return mock_client


@pytest.fixture
def mock_httpx_client():
    """
    Mocks the httpx.Client to prevent real network requests for URL processing.
    """
    # We patch the client where it's used in the extractors module.
    yield MagicMock()


@pytest.fixture
def mock_get_mime_type():
    """Deprecated shim: MIME detection handled in Source.from_file()."""
    m = MagicMock()
    m.side_effect = lambda *_a, **_k: "application/octet-stream"
    yield m


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
