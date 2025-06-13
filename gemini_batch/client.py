"""
Gemini API client for batch processing experiments
"""

from collections import deque
import time
from typing import Any, Dict, List, Optional, Union
import warnings

from google import genai
from google.genai import types

from .config import APITier, ConfigManager
from .exceptions import APIError, MissingKeyError, NetworkError
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
                    "Consider using ConfigManager exclusively for cleaner configuration.",
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
                "API key required. Provide via parameter, ConfigManager, or GEMINI_API_KEY environment variable."
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
            self.rate_limit_requests = 15
            self.rate_limit_tokens = 250_000

        self.rate_limit_window = 60  # seconds
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

    def _api_call_with_retry(self, api_call_func, max_retries: int = 2):
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
                        wait_time = (2**attempt) * 5  # Exponential backoff
                        print(
                            f"Rate limit hit. Retrying in {wait_time}s... "
                            f"(attempt {attempt + 1}/{max_retries + 1})"
                        )
                        time.sleep(wait_time)
                        continue

                # For other errors, retry with shorter delay
                elif attempt < max_retries:
                    wait_time = 2**attempt
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

            if return_usage:
                return {
                    "text": response.text,
                    "usage": extract_usage_metrics(response),
                }
            else:
                return response.text

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
