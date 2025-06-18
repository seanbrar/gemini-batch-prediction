"""
Gemini API client for batch processing experiments
"""

from collections import deque
import contextlib
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union
import warnings

from google import genai
from google.genai import types

from .config import APITier, ConfigManager
from .constants import (
    FALLBACK_REQUESTS_PER_MINUTE,
    FALLBACK_TOKENS_PER_MINUTE,
    FILE_POLL_INTERVAL,
    FILE_PROCESSING_TIMEOUT,
    FILES_API_THRESHOLD,
    MAX_RETRIES,
    RATE_LIMIT_RETRY_DELAY,
    RATE_LIMIT_WINDOW,
    RETRY_BASE_DELAY,
)
from .exceptions import APIError, MissingKeyError, NetworkError
from .files.utils import get_mime_type
from .utils import extract_usage_metrics


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
        """Initialize client with flexible configuration options

        Args:
            api_key: Explicit API key (overrides config and environment)
            model_name: Explicit model name (overrides config and environment)
            enable_caching: Enable caching for requests
            config_manager: Pre-configured ConfigManager instance
            tier: API tier for rate limiting (creates config if manager not provided)
            **kwargs: Additional arguments (for future extensibility)
        """

        # Configuration resolution with clean precedence
        if config_manager is not None:
            # Use provided config manager
            self.config = config_manager

            # Allow explicit parameters to override config values
            self.api_key = api_key or config_manager.api_key
            self.model_name = model_name or config_manager.model

            # Warn if both config and parameters provided
            if api_key or model_name or tier:
                warnings.warn(
                    "Explicit parameters override ConfigManager values. "
                    "Consider using ConfigManager exclusively for cleaner "
                    "configuration.",
                    stacklevel=2,
                )

        else:
            # Create config manager from parameters + environment
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
            # Fallback to conservative defaults if config fails
            self.rate_limit_requests = FALLBACK_REQUESTS_PER_MINUTE
            self.rate_limit_tokens = FALLBACK_TOKENS_PER_MINUTE

        self.rate_limit_window = RATE_LIMIT_WINDOW  # seconds
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
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Unified response processing with consistent usage tracking"""
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

    def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """Generate content from a single prompt with optional system instruction"""

        def api_call():
            if system_instruction:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction
                    ),
                    contents=prompt,
                )
            else:
                response = self.client.models.generate_content(
                    model=self.model_name, contents=prompt
                )

            return self._process_response(response, return_usage)

        try:
            return self._api_call_with_retry(api_call)
        except (ConnectionError, TimeoutError) as e:
            raise NetworkError(f"Network connection failed: {e}") from e
        except Exception as e:
            raise APIError(f"API call failed: {e}") from e

    def generate_batch(
        self, content: str, questions: List[str], return_usage: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """Generate batch response for multiple questions about content"""
        batch_prompt = self._create_batch_prompt(content, questions)
        return self.generate_content(batch_prompt, return_usage=return_usage)

    def _create_batch_prompt(self, content: str, questions: List[str]) -> str:
        """Create optimized batch prompt for multiple questions"""
        prompt = (
            f"Content: {content}\n\nPlease answer each of the following questions:\n\n"
        )

        for i, question in enumerate(questions, 1):
            prompt += f"Question {i}: {question}\n"

        prompt += "\nProvide numbered answers in this format:\n"
        for i in range(1, len(questions) + 1):
            prompt += f"Answer {i}: [Your response]\n"

        return prompt

    def upload_file(self, file_path, auto_cleanup: bool = True):
        """Upload file to Gemini Files API with processing wait"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise APIError(f"File not found: {file_path}")

        uploaded_file = self.client.files.upload(file=str(file_path))
        self._wait_for_file_processing(uploaded_file)
        return uploaded_file

    def generate_content_with_file(
        self,
        file_path,
        prompt: str,
        auto_cleanup: bool = True,
        return_usage: bool = False,
    ):
        """Generate content with file using appropriate method based on size"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise APIError(f"File not found: {file_path}")

        # Check file size to determine processing method
        file_size = file_path.stat().st_size

        # Files API threshold as per Gemini Files API documentation
        if file_size < FILES_API_THRESHOLD:
            return self._generate_content_inline(file_path, prompt, return_usage)
        else:
            return self._generate_content_files_api(
                file_path, prompt, auto_cleanup, return_usage
            )

    def _generate_content_inline(
        self, file_path, prompt: str, return_usage: bool = False
    ):
        """Generate content using inline file data (for files <20MB)"""
        mime_type = get_mime_type(file_path, use_magic=False)

        try:

            def api_call():
                # Read file as bytes for inline processing
                file_data = file_path.read_bytes()

                # Create inline part
                file_part = types.Part.from_bytes(data=file_data, mime_type=mime_type)

                response = self.client.models.generate_content(
                    model=self.model_name, contents=[file_part, prompt]
                )

                extra_metadata = {
                    "processing_method": "inline",
                    "file_size_mb": round(len(file_data) / (1024 * 1024), 2),
                }
                return self._process_response(response, return_usage, extra_metadata)

            return self._api_call_with_retry(api_call)

        except (ConnectionError, TimeoutError) as e:
            raise NetworkError(f"Network connection failed: {e}") from e
        except Exception as e:
            raise APIError(f"Inline multimodal generation failed: {e}") from e

    def _generate_content_files_api(
        self,
        file_path,
        prompt: str,
        auto_cleanup: bool = True,
        return_usage: bool = False,
    ):
        """Generate content using Files API (for files >=20MB)"""
        uploaded_file = self.upload_file(file_path)

        try:

            def api_call():
                response = self.client.models.generate_content(
                    model=self.model_name, contents=[uploaded_file, prompt]
                )

                extra_metadata = {
                    "processing_method": "files_api",
                    "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                }
                return self._process_response(response, return_usage, extra_metadata)

            result = self._api_call_with_retry(api_call)

            if auto_cleanup:
                with contextlib.suppress(Exception):
                    self.client.files.delete(name=uploaded_file.name)

            return result

        except (ConnectionError, TimeoutError) as e:
            raise NetworkError(f"Network connection failed: {e}") from e
        except Exception as e:
            raise APIError(f"Files API multimodal generation failed: {e}") from e

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
