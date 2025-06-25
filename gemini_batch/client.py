"""
Gemini API client with unified multimodal content processing
"""

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union

from google import genai
from google.genai import types
import httpx

from .config import APITier, ConfigManager
from .constants import (
    FALLBACK_REQUESTS_PER_MINUTE,
    FALLBACK_TOKENS_PER_MINUTE,
    FILE_POLL_INTERVAL,
    FILE_PROCESSING_TIMEOUT,
    MAX_RETRIES,
    RATE_LIMIT_RETRY_DELAY,
    RATE_LIMIT_WINDOW,
    RETRY_BASE_DELAY,
)
from .efficiency import extract_usage_metrics
from .exceptions import APIError, MissingKeyError
from .files import FileOperations
from .response import validate_structured_response


@dataclass
class ClientConfiguration:
    """Unified client configuration with type safety"""

    api_key: str
    model_name: str
    enable_caching: bool = False
    tier: Optional[APITier] = None
    custom_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config_manager(
        cls, config_manager: ConfigManager, **overrides
    ) -> "ClientConfiguration":
        """Create configuration from ConfigManager with optional overrides"""
        return cls(
            api_key=overrides.get("api_key") or config_manager.api_key,
            model_name=overrides.get("model_name") or config_manager.model,
            tier=overrides.get("tier"),
            enable_caching=overrides.get("enable_caching", False),
            custom_options=overrides.get("custom_options", {}),
        )

    @classmethod
    def from_parameters(
        cls,
        api_key: str = None,
        model_name: str = None,
        tier: Optional[APITier] = None,
        **kwargs,
    ) -> "ClientConfiguration":
        """Create configuration from individual parameters"""
        # Create ConfigManager to handle defaults and environment
        config_manager = ConfigManager(tier=tier, model=model_name, api_key=api_key)

        return cls(
            api_key=api_key or config_manager.api_key,
            model_name=model_name or config_manager.model,
            tier=tier,
            enable_caching=kwargs.get("enable_caching", False),
            custom_options={k: v for k, v in kwargs.items() if k != "enable_caching"},
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for debugging"""
        return asdict(self)

    def validate(self) -> None:
        """Validate configuration completeness"""
        if not self.api_key:
            raise MissingKeyError(
                "API key required. Provide via parameter, ConfigManager, "
                "or GEMINI_API_KEY environment variable."
            )


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""

    requests_per_minute: int
    tokens_per_minute: int
    window_seconds: int = RATE_LIMIT_WINDOW


class RateLimiter:
    """Handles rate limiting with context manager support"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_timestamps = deque()

    @contextmanager
    def request_context(self):
        """Context manager for rate-limited requests"""
        self._wait_if_needed()
        try:
            yield
        finally:
            self._record_request()

    def _wait_if_needed(self):
        """Wait if we're approaching the rate limit"""
        now = time.time()

        # Remove timestamps older than the rate limit window
        while (
            self.request_timestamps
            and now - self.request_timestamps[0] > self.config.window_seconds
        ):
            self.request_timestamps.popleft()

        # If we're at the limit, wait for the oldest request to age out
        if len(self.request_timestamps) >= self.config.requests_per_minute:
            sleep_time = (
                self.config.window_seconds - (now - self.request_timestamps[0]) + 1
            )
            if sleep_time > 0:
                print(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                # Clean up timestamps after waiting
                now = time.time()
                while (
                    self.request_timestamps
                    and now - self.request_timestamps[0] > self.config.window_seconds
                ):
                    self.request_timestamps.popleft()

    def _record_request(self):
        """Record a request timestamp"""
        self.request_timestamps.append(time.time())


class PromptBuilder:
    """Handles all prompt construction strategies"""

    def create_batch_prompt(
        self, questions: List[str], response_schema: Optional[Any] = None
    ) -> str:
        """Create batch prompt for multiple questions"""
        if response_schema:
            prompt = "Please answer each of the following questions. "
            prompt += "Your response will be automatically formatted according to the specified schema.\n\n"

            for i, question in enumerate(questions, 1):
                prompt += f"Question {i}: {question}\n"

            prompt += "\nProvide comprehensive answers for each question."
        else:
            prompt = "Please answer each of the following questions:\n\n"

            for i, question in enumerate(questions, 1):
                prompt += f"Question {i}: {question}\n"

            prompt += "\nProvide numbered answers in this format:\n"
            for i in range(1, len(questions) + 1):
                prompt += f"Answer {i}: [Your response]\n"

        return prompt

    def create_single_prompt(self, prompt: str) -> str:
        """Create single prompt (passthrough for consistency)"""
        return prompt


class ContentPartBuilder:
    """Handles complex part creation logic with clear separation of concerns"""

    def __init__(self, client: "GeminiClient"):
        self.client = client

    def build_parts(
        self, extracted_content, source_url: str = None
    ) -> List[types.Part]:
        """Build parts with strategy-based approach"""
        if extracted_content.requires_api_upload:
            return self._build_upload_parts(extracted_content, source_url)
        else:
            return self._build_inline_parts(extracted_content, source_url)

    def _build_upload_parts(
        self, extracted_content, source_url: str = None
    ) -> List[types.Part]:
        """Handle API upload strategy"""
        if extracted_content.file_path:
            # File path - upload to Files API
            uploaded_file = self.client._upload_file(extracted_content.file_path)
            return [types.Part(file_data=types.FileData(file_uri=uploaded_file.uri))]
        elif source_url:
            # URL content that needs API upload - not yet supported
            raise APIError(
                f"URL content too large for inline processing: {source_url}. "
                "Large URL content upload not yet supported."
            )
        else:
            raise APIError("Content requires API upload but no file path available")

    def _build_inline_parts(
        self, extracted_content, source_url: str = None
    ) -> List[types.Part]:
        """Handle inline processing strategy"""
        content_bytes = self._get_content_bytes(extracted_content, source_url)
        return [
            types.Part.from_bytes(
                data=content_bytes,
                mime_type=extracted_content.metadata["mime_type"],
            )
        ]

    def _get_content_bytes(self, extracted_content, source_url: str = None) -> bytes:
        """Handle content retrieval (file vs URL vs cache)"""
        if extracted_content.file_path:
            # File path - read file data
            return extracted_content.file_path.read_bytes()
        elif source_url:
            return self._get_url_content_bytes(extracted_content, source_url)
        else:
            raise APIError("No data source available for inline processing")

    def _get_url_content_bytes(self, extracted_content, source_url: str) -> bytes:
        """Get content bytes from URL with caching support"""
        if extracted_content.metadata.get("content_cached", False):
            # Get cached content from URLExtractor
            file_ops = FileOperations()
            url_extractor = None
            for extractor in file_ops.extractor_manager.extractors:
                if hasattr(extractor, "get_cached_content"):
                    url_extractor = extractor
                    break

            if url_extractor:
                file_data = url_extractor.get_cached_content(source_url)
                if file_data is None:
                    # Cache miss - download using extractor's method
                    file_data = url_extractor.download_content_if_needed(source_url)
                return file_data
            else:
                # Fallback to direct download
                return self._download_url_content(source_url)
        else:
            # Content not cached - download it
            return self._download_url_content(source_url)

    def _download_url_content(self, source_url: str) -> bytes:
        """Direct URL content download"""
        with httpx.Client(timeout=30) as client:
            response = client.get(source_url)
            response.raise_for_status()
            return response.content


class GenerationErrorHandler:
    """Centralized error handling with context awareness"""

    def handle_generation_error(
        self,
        error: Exception,
        response_schema: Optional[Any] = None,
        content_type: str = "unknown",
    ) -> None:
        """Handle with rich context for better error messages"""
        error_str = str(error).lower()

        # Structured output specific errors
        if response_schema and ("json" in error_str or "schema" in error_str):
            raise APIError(
                f"Structured output generation failed for {content_type} content. "
                f"Check your schema definition. Original error: {error}"
            ) from error

        # Content-specific errors
        if "quota" in error_str and "youtube" in error_str:
            raise APIError(
                f"YouTube quota exceeded. Free tier allows 8 hours/day. "
                f"Original error: {error}"
            ) from error
        elif "private" in error_str or "unlisted" in error_str:
            raise APIError(
                f"YouTube video must be public. Private/unlisted videos not supported. "
                f"Original error: {error}"
            ) from error
        elif "not found" in error_str or "unavailable" in error_str:
            raise APIError(
                f"Content not found or unavailable ({content_type}). "
                f"Original error: {error}"
            ) from error

        raise APIError(
            f"Content generation failed for {content_type}: {error}"
        ) from error


class ContentProcessor:
    """Pure interface: files subsystem → Gemini Parts coordination"""

    def __init__(self, client: "GeminiClient"):
        self.client = client
        self.file_ops = FileOperations()
        self.part_builder = ContentPartBuilder(client)
        self.strategies = {
            "direct_text": self._handle_text_content,
            "youtube_api": self._handle_youtube_content,
            "pdf_url_api": self._handle_pdf_url_content,
            "directory_scan": self._handle_directory_content,
        }

    def process_content(
        self, source: Union[str, Path, List[Union[str, Path]]]
    ) -> List[types.Part]:
        """Process any content type using appropriate strategy"""
        if isinstance(source, list):
            parts = []
            for item in source:
                parts.extend(self._process_single_source(item))
            return parts
        else:
            return self._process_single_source(source)

    def _process_single_source(self, source: Union[str, Path]) -> List[types.Part]:
        """Process a single source using strategy pattern"""
        try:
            extracted_content = self.file_ops.process_source(source)
            strategy = self.strategies.get(extracted_content.extraction_method)

            if strategy:
                return strategy(extracted_content)
            else:
                return self._handle_generic_content(extracted_content)

        except Exception as e:
            raise APIError(f"Failed to process source {source}: {e}") from e

    def _handle_text_content(self, extracted_content) -> List[types.Part]:
        """Handle direct text content"""
        return [types.Part(text=extracted_content.content)]

    def _handle_youtube_content(self, extracted_content) -> List[types.Part]:
        """Handle YouTube URL content"""
        youtube_url = extracted_content.metadata["url"]
        return [types.Part(file_data=types.FileData(file_uri=youtube_url))]

    def _handle_pdf_url_content(self, extracted_content) -> List[types.Part]:
        """Handle PDF URL content"""
        pdf_url = extracted_content.metadata["url"]
        return [types.Part(file_data=types.FileData(file_uri=pdf_url))]

    def _handle_directory_content(self, extracted_content) -> List[types.Part]:
        """Handle directory content"""
        return self._handle_directory_from_extracted(extracted_content)

    def _handle_generic_content(self, extracted_content) -> List[types.Part]:
        """Handle generic file or URL content"""
        return self.part_builder.build_parts(
            extracted_content, extracted_content.metadata.get("url")
        )

    def _handle_directory_from_extracted(self, extracted_content) -> List[types.Part]:
        """Handle directory processing using extracted content metadata"""
        directory_path = Path(extracted_content.metadata["directory_path"])

        try:
            # Use existing directory scanning infrastructure
            categorized_files = self.file_ops.scan_directory(directory_path)

            # Flatten all files into a single list
            all_files = []
            for file_type, files in categorized_files.items():
                all_files.extend(files)

            if not all_files:
                # Return a text part indicating empty directory
                return [
                    types.Part(
                        text=f"Directory {directory_path} contains no processable files."
                    )
                ]

            # Process each file through the files subsystem
            parts = []
            for file_info in all_files:
                try:
                    file_extracted = self.file_ops.extract_content(file_info.path)
                    file_parts = self.part_builder.build_parts(file_extracted)
                    parts.extend(file_parts)
                except Exception as e:
                    # Add error information as text part for failed files
                    parts.append(
                        types.Part(text=f"Error processing {file_info.path}: {e}")
                    )

            return parts

        except Exception as e:
            raise APIError(f"Failed to process directory {directory_path}: {e}") from e


class GeminiClient:
    """Streamlined Gemini API client with clear separation of concerns"""

    def __init__(self, config: ClientConfiguration):
        """Initialize client with unified configuration"""
        self.config = config
        self.config.validate()

        # Set up components
        self.client = genai.Client(api_key=config.api_key)
        self.rate_limiter = RateLimiter(self._get_rate_limit_config())
        self.prompt_builder = PromptBuilder()
        self.error_handler = GenerationErrorHandler()

    def _get_rate_limit_config(self) -> RateLimitConfig:
        """Get rate limiting configuration"""
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
        """Create client from ConfigManager with optional overrides"""
        client_config = ClientConfiguration.from_config_manager(
            config_manager, **overrides
        )
        return cls(client_config)

    @classmethod
    def from_env(cls, **overrides) -> "GeminiClient":
        """Create client from environment variables"""
        config_manager = ConfigManager.from_env()
        return cls.from_config_manager(config_manager, **overrides)

    @classmethod
    def with_defaults(cls, api_key: str, **overrides) -> "GeminiClient":
        """Create client with sensible defaults"""
        client_config = ClientConfiguration.from_parameters(
            api_key=api_key,
            model_name=overrides.get("model_name", "gemini-1.5-flash"),
            **overrides,
        )
        return cls(client_config)

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration for debugging"""
        summary = self.config.get_summary()
        summary.update(
            {
                "rate_limiter_config": {
                    "requests_per_minute": self.rate_limiter.config.requests_per_minute,
                    "tokens_per_minute": self.rate_limiter.config.tokens_per_minute,
                    "window_seconds": self.rate_limiter.config.window_seconds,
                }
            }
        )
        return summary

    def generate_content(
        self,
        content: Union[str, Path, List[Union[str, Path]]],
        prompt: Optional[str] = None,
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """Generate content from any source type"""
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
        """Process multiple questions about any content type"""
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
        """Unified generation method eliminating duplication"""
        # Prepare content
        content_processor = ContentProcessor(self)
        parts = content_processor.process_content(content)

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

    def _generate_with_parts(
        self,
        parts: List[types.Part],
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        **options,
    ) -> Union[str, Dict[str, Any]]:
        """Core generation method - single point of API interaction"""

        def api_call():
            config_params = {}

            if system_instruction:
                config_params["system_instruction"] = system_instruction

            if response_schema:
                config_params.update(
                    {
                        "response_mime_type": "application/json",
                        "response_schema": response_schema,
                    }
                )

            config = (
                types.GenerateContentConfig(**config_params) if config_params else None
            )

            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=types.Content(parts=parts),
                config=config,
            )

            return self._process_response(response, return_usage, response_schema)

        return self._api_call_with_retry(api_call)

    def _get_content_type_description(self, content) -> str:
        """Get human-readable description of content type for error messages"""
        if isinstance(content, list):
            return f"mixed content ({len(content)} items)"
        elif isinstance(content, Path):
            return f"file ({content.suffix or 'unknown type'})"
        elif isinstance(content, str):
            if content.startswith(("http://", "https://")):
                if "youtube.com" in content or "youtu.be" in content:
                    return "YouTube video"
                else:
                    return "URL content"
            else:
                return "text content"
        else:
            return "unknown content"

    def _process_response(
        self,
        response,
        return_usage: bool = False,
        response_schema: Optional[Any] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Unified response processing with consistent usage tracking"""

        # Handle structured output with validation
        if response_schema:
            # Try to use original parsed data from API first (highest fidelity)
            if hasattr(response, "parsed") and response.parsed is not None:
                # API provided structured data - use it directly
                parsed_data = response.parsed
                structured_success = True
                structured_confidence = 1.0
                validation_method = "api_native"
                validation_errors = []
            else:
                # Fallback to validation logic for text parsing
                validation_result = validate_structured_response(
                    response, response_schema
                )
                parsed_data = validation_result.parsed_data
                structured_success = validation_result.success
                structured_confidence = validation_result.confidence
                validation_method = validation_result.validation_method
                validation_errors = validation_result.errors

            if return_usage:
                result = {
                    "text": response.text,
                    "parsed": parsed_data,
                    "structured_success": structured_success,
                    "structured_confidence": structured_confidence,
                    "validation_method": validation_method,
                    "validation_errors": validation_errors,
                    "usage": extract_usage_metrics(response),
                }
                if extra_metadata:
                    result.update(extra_metadata)
                return result
            else:
                # Return best available data
                return parsed_data if structured_success else response.text

        # Standard text response processing
        if return_usage:
            result = {
                "text": response.text,
                "usage": extract_usage_metrics(response),
            }
            if extra_metadata:
                result.update(extra_metadata)
            return result
        else:
            return response.text

    def _api_call_with_retry(
        self, api_call_func: Callable, max_retries: int = MAX_RETRIES
    ):
        """Execute API call with simplified retry logic"""
        for attempt in range(max_retries + 1):
            try:
                return api_call_func()

            except Exception as e:
                if attempt < max_retries:
                    error_str = str(e).lower()

                    # Check if it's a rate limit error
                    if (
                        "rate limit" in error_str
                        or "quota" in error_str
                        or "429" in error_str
                    ):
                        wait_time = (2**attempt) * RATE_LIMIT_RETRY_DELAY
                        print(
                            f"Rate limit hit. Retrying in {wait_time}s... "
                            f"(attempt {attempt + 1}/{max_retries + 1})"
                        )
                    else:
                        wait_time = (2**attempt) * RETRY_BASE_DELAY
                        print(
                            f"API error. Retrying in {wait_time}s... "
                            f"(attempt {attempt + 1}/{max_retries + 1})"
                        )

                    time.sleep(wait_time)
                    continue

                # Final attempt failed
                raise

    def _upload_file(self, file_path: Path):
        """Upload file to Gemini Files API with processing wait"""
        uploaded_file = self.client.files.upload(file=str(file_path))
        self._wait_for_file_processing(uploaded_file)
        return uploaded_file

    def _wait_for_file_processing(
        self, uploaded_file, timeout: int = FILE_PROCESSING_TIMEOUT
    ):
        """Wait for file processing to complete"""
        start_time = time.time()
        poll_interval = FILE_POLL_INTERVAL

        while uploaded_file.state.name == "PROCESSING":
            if time.time() - start_time > timeout:
                raise APIError(f"File processing timeout: {uploaded_file.display_name}")

            time.sleep(poll_interval)
            uploaded_file = self.client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise APIError(f"File processing failed: {uploaded_file.display_name}")
