"""
Test adapter for transitioning from old BatchProcessor to new GeminiExecutor architecture.

This adapter allows existing characterization tests to work with the new architecture
without requiring changes to the test code or golden files.
"""

import asyncio
from typing import Any

from gemini_batch.core.types import InitialCommand
from gemini_batch.executor import GeminiExecutor


class TestAdapterBatchProcessor:
    """
    An adapter that presents the old BatchProcessor interface to tests,
    but executes using the new pipeline architecture.

    This allows existing characterization tests to work with the new architecture
    without requiring changes to the test code or golden files.
    """

    def __init__(self, **config_overrides):
        """Initialize the adapter with configuration overrides.

        Args:
            **config_overrides: Configuration parameters to override defaults
        """
        # Create a config object from overrides, with sensible defaults
        config_dict = {
            "api_key": config_overrides.get("api_key", "mock_api_key_for_tests"),
            "model": config_overrides.get("model", "gemini-2.0-flash"),
            "enable_caching": config_overrides.get("enable_caching", False),
        }

        # Add any additional config overrides
        config_dict.update(config_overrides)

        # Build a config dictionary compatible with the library (runtime-typed)
        config_typed: dict[str, Any] = {
            "api_key": config_dict.get("api_key", "mock_api_key_for_tests"),
            "model": config_dict.get("model", "gemini-2.0-flash"),
            "enable_caching": config_dict.get("enable_caching", False),
        }
        if "tier" in config_dict:
            # Preserve optional tier when present; typing relaxed in tests
            config_typed["tier"] = config_dict["tier"]

        # Cast to the library's expected TypedDict (acceptable in test adapter)
        self.config = config_typed
        self.executor = GeminiExecutor(config=self.config)  # type: ignore[arg-type]

    def process_questions(
        self,
        content: str | list[str],
        questions: list[str],
        _compare_methods: bool = False,  # noqa: FBT001, FBT002
        _response_schema: Any | None = None,
        _return_usage: bool = False,  # noqa: FBT001, FBT002
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Mimics the old BatchProcessor.process_questions method signature.

        Args:
            content: Content to process (text, files, URLs, etc.)
            questions: List of questions to ask
            compare_methods: Whether to run individual comparison (not yet supported in new arch)
            response_schema: Optional schema for structured output
            return_usage: Whether to include usage metrics
            **kwargs: Additional parameters (currently ignored)

        Returns:
            Dictionary with the same structure as the old BatchProcessor output
        """
        # Convert content to tuple format expected by InitialCommand
        sources = tuple(content) if isinstance(content, list) else (content,)

        # Convert questions to tuple format
        prompts = tuple(questions)

        # Create the command for the new architecture
        command = InitialCommand(
            sources=sources,
            prompts=prompts,
            config=self.config,  # type: ignore[arg-type]
        )

        # Execute using the new async executor
        # Since our tests are synchronous, we run the async method here
        try:
            return asyncio.run(self.executor.execute(command))
        except Exception as e:
            # Convert exceptions to the format expected by tests
            return {
                "success": False,
                "error": str(e),
                "answers": [],
                "question_count": len(questions),
                "efficiency": {},
                "metrics": {"batch": {}, "individual": {}},
                "processing_time": 0.0,
            }

    def process_questions_multi_source(
        self,
        sources: list[str | list[str]],
        questions: list[str],
        response_schema: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Mimics the old BatchProcessor.process_questions_multi_source method.

        Args:
            sources: List of content sources to process
            questions: List of questions to ask
            response_schema: Optional schema for structured output
            **kwargs: Additional parameters

        Returns:
            Dictionary with the same structure as the old BatchProcessor output
        """
        # Flatten sources if they're nested
        flat_sources = []
        for source in sources:
            if isinstance(source, list):
                flat_sources.extend(source)
            else:
                flat_sources.append(source)

        # Use the single-source method with flattened content
        return self.process_questions(
            content=flat_sources,
            questions=questions,
            response_schema=response_schema,
            **kwargs,
        )
