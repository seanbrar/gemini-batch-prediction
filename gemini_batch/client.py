"""
Gemini API client for batch processing experiments
"""

from collections import deque
import os
import time
from typing import Any, Dict, List, Optional, Union

from google import genai
from google.genai import types

from .exceptions import APIError, MissingKeyError, NetworkError
from .utils import extract_usage_metrics


class GeminiClient:
    """Gemini API client with batch processing capabilities"""

    def __init__(
        self, api_key: str = None, model_name: str = None, enable_caching: bool = False
    ):
        """Initialize client with API key, model, and rate limiting"""
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise MissingKeyError(
                "API key required. Set GEMINI_API_KEY environment variable."
            )

        if model_name is None:
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.enable_caching = enable_caching

        # Basic rate limiting for 2.0 Flash free tier (15 requests/minute)
        self.rate_limit_requests = 15
        self.rate_limit_window = 60  # seconds
        self.request_timestamps = deque()

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
