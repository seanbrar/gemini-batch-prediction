"""
Gemini API client with unified multimodal content processing
"""

from collections import deque
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union
import warnings

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


class ContentSource:
    """Thin orchestration layer that delegates to the files subsystem"""

    def __init__(
        self, source: Union[str, Path, List[Union[str, Path]]], client: "GeminiClient"
    ):
        self.source = source
        self.client = client
        self.file_ops = FileOperations()
        self.parts = self._prepare_parts()

    def _prepare_parts(self) -> List[types.Part]:
        """Convert any source type into Gemini API Parts"""
        if isinstance(self.source, list):
            parts = []
            for item in self.source:
                parts.extend(self._process_single_source(item))
            return parts
        else:
            return self._process_single_source(self.source)

    def _process_single_source(self, source: Union[str, Path]) -> List[types.Part]:
        """Process a single source into Part(s)"""
        try:
            # Delegate all source processing to FileOperations
            extracted_content = self.file_ops.process_source(source)

            # Handle different extraction methods
            if extracted_content.extraction_method == "direct_text":
                # Text content - create text part
                return [types.Part(text=extracted_content.content)]

            elif extracted_content.extraction_method == "youtube_api":
                # YouTube URL - create file data part with URL
                youtube_url = extracted_content.metadata["url"]
                return [types.Part(file_data=types.FileData(file_uri=youtube_url))]

            elif extracted_content.extraction_method == "pdf_url_api":
                # Large PDF URL - create file data part with URL (like YouTube)
                pdf_url = extracted_content.metadata["url"]
                return [types.Part(file_data=types.FileData(file_uri=pdf_url))]

            elif extracted_content.extraction_method == "directory_scan":
                # Directory - scan and process all files
                return self._handle_directory_from_extracted(extracted_content)

            else:
                # File or URL extraction - create parts from extracted content
                return self._create_parts_from_extracted(
                    extracted_content, extracted_content.metadata.get("url")
                )

        except Exception as e:
            raise APIError(f"Failed to process source {source}: {e}") from e

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
                    file_parts = self._create_parts_from_extracted(file_extracted)
                    parts.extend(file_parts)
                except Exception as e:
                    # Add error information as text part for failed files
                    parts.append(
                        types.Part(text=f"Error processing {file_info.path}: {e}")
                    )

            return parts

        except Exception as e:
            raise APIError(f"Failed to process directory {directory_path}: {e}") from e

    def _create_parts_from_extracted(
        self, extracted_content, source_url: str = None
    ) -> List[types.Part]:
        """Create API Parts from ExtractedContent, handling both files and URLs"""
        if extracted_content.requires_api_upload:
            if extracted_content.file_path:
                # File path - upload to Files API
                uploaded_file = self.client._upload_file(extracted_content.file_path)
                return [
                    types.Part(file_data=types.FileData(file_uri=uploaded_file.uri))
                ]
            elif source_url:
                # URL content that needs API upload - we need the raw data
                # For now, URLs always use inline processing to avoid complexity
                # This could be enhanced later to support large URL uploads
                raise APIError(
                    f"URL content too large for inline processing: {source_url}. "
                    "Large URL content upload not yet supported."
                )
            else:
                raise APIError("Content requires API upload but no file path available")
        else:
            # Inline processing - create Part from file data or URL data
            if extracted_content.file_path:
                # File path - read file data
                file_data = extracted_content.file_path.read_bytes()
            elif source_url:
                # URL - check if content is cached to avoid double download
                if extracted_content.metadata.get("content_cached", False):
                    # Get cached content from URLExtractor
                    url_extractor = None
                    for extractor in self.file_ops.extractor_manager.extractors:
                        if hasattr(extractor, "get_cached_content"):
                            url_extractor = extractor
                            break

                    if url_extractor:
                        file_data = url_extractor.get_cached_content(source_url)
                        if file_data is None:
                            # Cache miss - download using extractor's method
                            file_data = url_extractor.download_content_if_needed(
                                source_url
                            )
                    else:
                        # Fallback to direct download
                        with httpx.Client(timeout=30) as client:
                            response = client.get(source_url)
                            response.raise_for_status()
                            file_data = response.content
                else:
                    # Content not cached - download it
                    with httpx.Client(timeout=30) as client:
                        response = client.get(source_url)
                        response.raise_for_status()
                        file_data = response.content
            else:
                raise APIError("No data source available for inline processing")

            return [
                types.Part.from_bytes(
                    data=file_data,
                    mime_type=extracted_content.metadata["mime_type"],
                )
            ]


class GeminiClient:
    """Gemini API client with batch processing capabilities"""

    def __init__(
        self,
        api_key: str = None,
        model_name: str = None,
        enable_caching: bool = False,
        config_manager: Optional[ConfigManager] = None,
        tier: Optional[APITier] = None,
        **kwargs,
    ):
        """Initialize client with flexible configuration options"""

        # Configuration resolution with clean precedence
        if config_manager is not None:
            self.config = config_manager
            self.api_key = api_key or config_manager.api_key
            self.model_name = model_name or config_manager.model

            if api_key or model_name or tier:
                warnings.warn(
                    "Explicit parameters override ConfigManager values. "
                    "Consider using ConfigManager exclusively for cleaner "
                    "configuration.",
                    stacklevel=2,
                )
        else:
            self.config = ConfigManager(tier=tier, model=model_name, api_key=api_key)
            self.api_key = self.config.api_key
            self.model_name = self.config.model

        # Validate we have an API key
        if not self.api_key:
            raise MissingKeyError(
                "API key required. Provide via parameter, ConfigManager, "
                "or GEMINI_API_KEY environment variable."
            )

        # Client-specific configuration
        self.enable_caching = enable_caching

        # Set up Google AI client
        self.client = genai.Client(api_key=self.api_key)

        # Set up rate limiting using config
        self._setup_rate_limiting()

    @classmethod
    def from_config(cls, config: ConfigManager, **kwargs) -> "GeminiClient":
        """Factory method for creating client from ConfigManager"""
        return cls(config_manager=config, **kwargs)

    @classmethod
    def from_env(cls, **kwargs) -> "GeminiClient":
        """Factory method for environment-driven configuration"""
        config = ConfigManager.from_env()
        return cls(config_manager=config, **kwargs)

    def _setup_rate_limiting(self):
        """Set up rate limiting using configuration"""
        try:
            rate_config = self.config.get_rate_limiter_config(self.model_name)
            self.rate_limit_requests = rate_config["requests_per_minute"]
            self.rate_limit_tokens = rate_config["tokens_per_minute"]
        except Exception:
            self.rate_limit_requests = FALLBACK_REQUESTS_PER_MINUTE
            self.rate_limit_tokens = FALLBACK_TOKENS_PER_MINUTE

        self.rate_limit_window = RATE_LIMIT_WINDOW
        self.request_timestamps = deque()

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration for debugging"""
        summary = self.config.get_config_summary()
        summary.update(
            {
                "client_model_name": self.model_name,
                "rate_limit_requests": self.rate_limit_requests,
                "rate_limit_tokens": getattr(self, "rate_limit_tokens", "unknown"),
                "enable_caching": self.enable_caching,
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
        # Prepare content parts
        content_source = ContentSource(content, self)
        parts = content_source.parts.copy()

        # Add prompt if provided
        if prompt:
            parts.append(types.Part(text=prompt))

        # Single unified API call
        return self._generate_with_parts(
            parts, system_instruction, return_usage, response_schema, **options
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

        # Prepare content parts
        content_source = ContentSource(content, self)
        parts = content_source.parts.copy()

        # Add batch prompt
        batch_prompt = self._create_batch_prompt(questions, response_schema)
        parts.append(types.Part(text=batch_prompt))

        # Single unified API call
        return self._generate_with_parts(
            parts, system_instruction, return_usage, response_schema, **options
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
                model=self.model_name,
                contents=types.Content(parts=parts),
                config=config,
            )

            return self._process_response(response, return_usage, response_schema)

        try:
            return self._api_call_with_retry(api_call)
        except Exception as e:
            # Enhanced error handling for content-specific issues
            error_str = str(e).lower()

            # Structured output specific errors
            if response_schema and ("json" in error_str or "schema" in error_str):
                raise APIError(
                    f"Structured output generation failed. Check your schema definition. "
                    f"Original error: {e}"
                ) from e

            if "quota" in error_str and "youtube" in error_str:
                raise APIError(
                    f"YouTube quota exceeded. Free tier allows 8 hours/day. "
                    f"Original error: {e}"
                ) from e
            elif "private" in error_str or "unlisted" in error_str:
                raise APIError(
                    f"YouTube video must be public. Private/unlisted videos not supported. "
                    f"Original error: {e}"
                ) from e
            elif "not found" in error_str or "unavailable" in error_str:
                raise APIError(
                    f"Content not found or unavailable. Original error: {e}"
                ) from e

            raise APIError(f"Content generation failed: {e}") from e

    def _create_batch_prompt(
        self, questions: List[str], response_schema: Optional[Any] = None
    ) -> str:
        """Create batch prompt for multiple questions, aware of structured output"""
        if response_schema:
            # For structured output, provide a more schema-friendly prompt
            prompt = "Please answer each of the following questions. "
            prompt += "Your response will be automatically formatted according to the specified schema.\n\n"

            for i, question in enumerate(questions, 1):
                prompt += f"Question {i}: {question}\n"

            prompt += "\nProvide comprehensive answers for each question."
        else:
            # Traditional text-based prompt
            prompt = "Please answer each of the following questions:\n\n"

            for i, question in enumerate(questions, 1):
                prompt += f"Question {i}: {question}\n"

            prompt += "\nProvide numbered answers in this format:\n"
            for i in range(1, len(questions) + 1):
                prompt += f"Answer {i}: [Your response]\n"

        return prompt

    def _wait_for_rate_limit(self):
        """Simple rate limiting: wait if we're approaching the limit"""
        now = time.time()

        # Remove timestamps older than the rate limit window
        while (
            self.request_timestamps
            and now - self.request_timestamps[0] > self.rate_limit_window
        ):
            self.request_timestamps.popleft()

        # If we're at the limit, wait for the oldest request to age out
        if len(self.request_timestamps) >= self.rate_limit_requests:
            sleep_time = self.rate_limit_window - (now - self.request_timestamps[0]) + 1
            if sleep_time > 0:
                print(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                # Clean up timestamps after waiting
                now = time.time()
                while (
                    self.request_timestamps
                    and now - self.request_timestamps[0] > self.rate_limit_window
                ):
                    self.request_timestamps.popleft()

        # Record this request
        self.request_timestamps.append(now)

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
        """Execute API call with basic retry logic"""
        for attempt in range(max_retries + 1):
            try:
                # Rate limiting before each attempt
                self._wait_for_rate_limit()

                # Make the API call
                return api_call_func()

            except Exception as e:
                error_str = str(e).lower()

                # Check if it's a rate limit error
                if (
                    "rate limit" in error_str
                    or "quota" in error_str
                    or "429" in error_str
                ):
                    if attempt < max_retries:
                        wait_time = (2**attempt) * RATE_LIMIT_RETRY_DELAY
                        print(
                            f"Rate limit hit. Retrying in {wait_time}s... "
                            f"(attempt {attempt + 1}/{max_retries + 1})"
                        )
                        time.sleep(wait_time)
                        continue

                # For other errors, retry with shorter delay
                elif attempt < max_retries:
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
