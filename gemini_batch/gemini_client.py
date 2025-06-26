"""
Main Gemini API client for content generation and batch processing
"""

from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union

from google import genai
from google.genai import types

from .client.configuration import ClientConfiguration, RateLimitConfig
from .client.content_processor import ContentProcessor
from .client.error_handler import GenerationErrorHandler
from .client.prompt_builder import PromptBuilder
from .client.rate_limiter import RateLimiter
from .client.token_counter import TokenCounter
from .config import ConfigManager
from .constants import (
    FALLBACK_REQUESTS_PER_MINUTE,
    FALLBACK_TOKENS_PER_MINUTE,
    FILE_POLL_INTERVAL,
    FILE_PROCESSING_TIMEOUT,
    MAX_RETRIES,
    RATE_LIMIT_RETRY_DELAY,
    RETRY_BASE_DELAY,
)
from .efficiency import extract_usage_metrics
from .exceptions import APIError
from .response import validate_structured_response


class GeminiClient:
    """Main Gemini API client for content generation and batch processing"""

    def __init__(self, config: ClientConfiguration):
        """Initialize client with configuration"""
        self.config = config
        self.config.validate()

        # Set up components
        self.client = genai.Client(api_key=config.api_key)
        self.rate_limiter = RateLimiter(self._get_rate_limit_config())
        self.prompt_builder = PromptBuilder()
        self.error_handler = GenerationErrorHandler()

        # Initialize caching components if enabled
        if config.enable_caching:
            self.config_manager = ConfigManager(
                tier=config.tier, model=config.model_name, api_key=config.api_key
            )
            self.token_counter = TokenCounter(self.client, self.config_manager)
        else:
            self.config_manager = None
            self.token_counter = None

        # Initialize file operations - pure file logic only
        from .files import FileOperations

        self.file_ops = FileOperations()

        # Initialize content processor for orchestration
        self.content_processor = ContentProcessor(self.file_ops)

    def _get_rate_limit_config(self) -> RateLimitConfig:
        """Get rate limiting configuration for the current model"""
        try:
            # Create temporary ConfigManager for rate limit config
            temp_config = ConfigManager(
                api_key=self.config.api_key,
                model=self.config.model_name,
                tier=self.config.tier,
            )
            rate_config = temp_config.get_rate_limiter_config(self.config.model_name)
            return RateLimitConfig(
                requests_per_minute=rate_config["requests_per_minute"],
                tokens_per_minute=rate_config["tokens_per_minute"],
            )
        except Exception:
            return RateLimitConfig(
                requests_per_minute=FALLBACK_REQUESTS_PER_MINUTE,
                tokens_per_minute=FALLBACK_TOKENS_PER_MINUTE,
            )

    @classmethod
    def from_config_manager(
        cls, config_manager: ConfigManager, **overrides
    ) -> "GeminiClient":
        """Create client from ConfigManager with optional parameter overrides"""
        client_config = ClientConfiguration.from_config_manager(
            config_manager, **overrides
        )
        return cls(client_config)

    @classmethod
    def from_env(cls, **overrides) -> "GeminiClient":
        """Create client from environment variables with optional overrides"""
        config_manager = ConfigManager.from_env()
        return cls.from_config_manager(config_manager, **overrides)

    @classmethod
    def with_defaults(cls, api_key: str, **overrides) -> "GeminiClient":
        """Create client with sensible defaults for quick setup"""
        client_config = ClientConfiguration.from_parameters(
            api_key=api_key,
            model_name=overrides.get("model_name", "gemini-2.0-flash"),
            **overrides,
        )
        return cls(client_config)

    def get_config_summary(self) -> Dict[str, Any]:
        """Get current configuration summary for debugging"""
        summary = self.config.get_summary()
        summary.update(
            {
                "rate_limiter_config": {
                    "requests_per_minute": self.rate_limiter.config.requests_per_minute,
                    "tokens_per_minute": self.rate_limiter.config.tokens_per_minute,
                    "window_seconds": self.rate_limiter.config.window_seconds,
                },
                "caching_enabled": self.config.enable_caching,
                "caching_available": self.token_counter is not None,
            }
        )
        if self.config_manager:
            summary["caching_thresholds"] = self.config_manager.get_caching_thresholds(
                self.config.model_name
            )
        return summary

    def should_cache_content(self, content, prefer_implicit: bool = True) -> bool:
        """
        Simple caching decision - returns True if content should be cached
        """
        if not self.token_counter:
            return False

        estimate = self.token_counter.estimate_for_caching(
            self.config.model_name, content, prefer_implicit
        )
        return estimate["cacheable"]

    def analyze_caching_strategy(
        self, content, prefer_implicit: bool = True
    ) -> Dict[str, Any]:
        """
        Detailed caching analysis with strategy recommendations
        """
        if not self.token_counter:
            return {
                "tokens": 0,
                "cacheable": False,
                "strategy": "none",
                "details": {"error": "Caching not enabled"},
                "caching_enabled": False,
            }

        estimate = self.token_counter.estimate_for_caching(
            self.config.model_name, content, prefer_implicit
        )

        return {
            "tokens": estimate["tokens"],
            "cacheable": estimate["cacheable"],
            "strategy": estimate["recommended_strategy"],
            "details": estimate,
            "caching_enabled": True,
        }

    def get_caching_thresholds(self) -> Dict[str, Optional[int]]:
        """Get caching thresholds for the current model"""
        if not self.config_manager:
            return {"implicit": None, "explicit": None}
        return self.config_manager.get_caching_thresholds(self.config.model_name)

    def generate_content(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        prompt: Optional[str] = None,
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """Generate content from text, files, URLs, or mixed sources"""
        return self._execute_generation(
            content=content,
            prompts=[prompt] if prompt else [],
            system_instruction=system_instruction,
            return_usage=return_usage,
            response_schema=response_schema,
            is_batch=False,
            **options,
        )

    def generate_batch(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        questions: List[str],
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """Process multiple questions about the same content"""
        if not questions:
            raise APIError("At least one question is required for batch processing")

        return self._execute_generation(
            content=content,
            prompts=questions,
            system_instruction=system_instruction,
            return_usage=return_usage,
            response_schema=response_schema,
            is_batch=True,
            **options,
        )

    def _execute_generation(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        prompts: List[str],
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        is_batch: bool = False,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """Core generation method for both single and batch requests"""
        # Get extracted content from orchestrator (pure file operations)
        extracted_contents = self.content_processor.process_content(content)

        # Convert to API parts (including uploads) - API responsibility
        parts = []
        for extracted in extracted_contents:
            part = self._create_api_part(extracted)
            parts.append(part)

        # Add prompts using PromptBuilder
        if is_batch:
            batch_prompt = self.prompt_builder.create_batch_prompt(
                prompts, response_schema
            )
            parts.append(types.Part(text=batch_prompt))
        else:
            for prompt in prompts:
                if prompt:
                    clean_prompt = self.prompt_builder.create_single_prompt(prompt)
                    parts.append(types.Part(text=clean_prompt))

        # Execute API call with rate limiting
        with self.rate_limiter.request_context():
            try:
                return self._generate_with_parts(
                    parts, system_instruction, return_usage, response_schema, **options
                )
            except Exception as e:
                # Determine content type for better error context
                content_type = self._get_content_type_description(content)
                self.error_handler.handle_generation_error(
                    e, response_schema, content_type
                )

    def _create_api_part(self, extracted_content) -> types.Part:
        """Convert extracted content to API part - handles upload if needed"""
        # Handle different extraction methods
        if extracted_content.extraction_method == "direct_text":
            return types.Part(text=extracted_content.content)
        elif extracted_content.extraction_method == "youtube_api":
            youtube_url = extracted_content.metadata["url"]
            return types.Part(file_data=types.FileData(file_uri=youtube_url))
        elif extracted_content.extraction_method == "pdf_url_api":
            pdf_url = extracted_content.metadata["url"]
            return types.Part(file_data=types.FileData(file_uri=pdf_url))
        elif extracted_content.extraction_method == "directory_scan":
            return self._handle_directory_extracted_content(extracted_content)
        elif extracted_content.extraction_method == "error":
            return types.Part(text=extracted_content.content)
        else:
            # Generic content - handle upload vs inline
            if extracted_content.requires_api_upload:
                return self._upload_file_if_needed(extracted_content)
            else:
                return self._create_inline_part(extracted_content)

    def _upload_file_if_needed(self, extracted_content) -> types.Part:
        """Handle upload when content requires it - API operation"""
        if not extracted_content.file_path:
            raise APIError("Content requires upload but no file path available")

        uploaded_file = self._upload_file(extracted_content.file_path)
        return types.Part(file_data=types.FileData(file_uri=uploaded_file.uri))

    def _create_inline_part(self, extracted_content) -> types.Part:
        """Create inline part from extracted content"""
        if extracted_content.file_path:
            # File path - read file data
            content_bytes = extracted_content.file_path.read_bytes()
        else:
            # Convert string content to bytes
            content_bytes = extracted_content.content.encode("utf-8")

        return types.Part.from_bytes(
            data=content_bytes,
            mime_type=extracted_content.metadata.get("mime_type", "text/plain"),
        )

    def _handle_directory_extracted_content(self, extracted_content) -> types.Part:
        """Handle directory content by processing individual files"""
        directory_path = Path(extracted_content.metadata["directory_path"])

        try:
            # Use existing directory scanning infrastructure
            categorized_files = self.file_ops.scan_directory(directory_path)

            # Flatten all files into a single list
            all_files = []
            for file_type, files in categorized_files.items():
                all_files.extend(files)

            if not all_files:
                return types.Part(
                    text=f"Directory {directory_path} contains no processable files."
                )

            # Process each file and combine into text summary
            file_contents = []
            for file_info in all_files:
                try:
                    file_extracted = self.file_ops.process_source(file_info.path)
                    if file_extracted.extraction_method == "direct_text":
                        file_contents.append(
                            f"=== {file_info.path.name} ===\n{file_extracted.content}\n"
                        )
                    else:
                        file_contents.append(
                            f"=== {file_info.path.name} ===\n[Binary file - {file_extracted.metadata.get('mime_type', 'unknown type')}]\n"
                        )
                except Exception as e:
                    file_contents.append(f"=== {file_info.path.name} ===\nError: {e}\n")

            combined_content = "\n".join(file_contents)
            return types.Part(text=combined_content)

        except Exception as e:
            raise APIError(f"Failed to process directory {directory_path}: {e}") from e

    def _upload_file(self, file_path: Path):
        """Upload file to Gemini Files API - API operation belongs here"""
        try:
            uploaded_file = self.client.files.upload(file=str(file_path))
            self._wait_for_file_processing(uploaded_file)
            return uploaded_file
        except Exception as e:
            raise APIError(f"Failed to upload file {file_path}: {e}") from e

    def _wait_for_file_processing(
        self, uploaded_file, timeout: int = FILE_PROCESSING_TIMEOUT
    ):
        """Wait for uploaded file to finish processing"""
        start_time = time.time()
        poll_interval = FILE_POLL_INTERVAL

        while uploaded_file.state.name == "PROCESSING":
            if time.time() - start_time > timeout:
                raise APIError(f"File processing timeout: {uploaded_file.display_name}")

            time.sleep(poll_interval)
            uploaded_file = self.client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise APIError(f"File processing failed: {uploaded_file.display_name}")

    def _generate_with_parts(
        self,
        parts: List[types.Part],
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """Execute API call with prepared content parts"""

        def api_call():
            # Build config parameters properly
            config_params = {}

            if system_instruction:
                config_params["system_instruction"] = system_instruction

            if response_schema:
                config_params["response_mime_type"] = "application/json"
                config_params["response_schema"] = response_schema

            # Create config only if we have parameters
            config = (
                types.GenerateContentConfig(**config_params) if config_params else None
            )

            return self.client.models.generate_content(
                model=self.config.model_name,
                contents=types.Content(parts=parts),  # Proper API wrapping
                config=config,
            )

        response = self._api_call_with_retry(api_call)

        return self._process_response(
            response, return_usage, response_schema, {"model": self.config.model_name}
        )

    def _get_content_type_description(self, content) -> str:
        """Get descriptive name for content type for error messages"""
        if isinstance(content, list):
            return f"mixed content ({len(content)} items)"
        elif isinstance(content, Path):
            return f"file ({content.suffix or 'no extension'})"
        elif isinstance(content, str):
            if content.startswith(("http://", "https://")):
                return "URL"
            elif len(content) < 100:
                return "text"
            else:
                return "long text"
        else:
            return "unknown content type"

    def _process_response(
        self,
        response,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Process API response with optional usage tracking and validation"""

        # Handle structured output with validation
        if response_schema:
            validation_result = validate_structured_response(response, response_schema)

            if return_usage:
                # Extract and return with usage information
                usage_info = extract_usage_metrics(response)

                result = {
                    "text": response.text,
                    "parsed": validation_result.parsed_data,
                    "structured_success": validation_result.success,
                    "structured_confidence": validation_result.confidence,
                    "validation_method": validation_result.validation_method,
                    "validation_errors": validation_result.errors,
                    "usage": usage_info,
                }

                # Add extra metadata if provided
                if extra_metadata:
                    result.update(extra_metadata)

                return result
            else:
                # Return best available data
                return (
                    validation_result.parsed_data
                    if validation_result.success
                    else response.text
                )

        # Standard text response
        if return_usage:
            # Extract and return with usage information
            usage_info = extract_usage_metrics(response)

            result = {
                "text": response.text,
                "usage": usage_info,
            }

            # Add extra metadata if provided
            if extra_metadata:
                result.update(extra_metadata)

            return result
        else:
            # Just return the text
            return response.text

    def _api_call_with_retry(
        self, api_call_func: Callable, max_retries: int = MAX_RETRIES
    ):
        """Execute API call with exponential backoff retry logic"""
        for attempt in range(max_retries + 1):
            try:
                return api_call_func()
            except Exception as error:
                if attempt == max_retries:
                    # Final attempt failed
                    raise

                # Check if it's a retryable error
                error_str = str(error).lower()
                if any(
                    retryable in error_str
                    for retryable in [
                        "rate limit",
                        "429",
                        "quota",
                        "timeout",
                        "temporary",
                        "service unavailable",
                        "500",
                        "502",
                        "503",
                        "504",
                    ]
                ):
                    # Calculate delay with exponential backoff
                    if "rate limit" in error_str or "429" in error_str:
                        delay = RATE_LIMIT_RETRY_DELAY
                    else:
                        delay = RETRY_BASE_DELAY * (2**attempt)

                    time.sleep(delay)
                    continue
                else:
                    # Non-retryable error
                    raise
