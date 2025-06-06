"""
Gemini API client for batch processing experiments
"""

import os
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
        """Initialize client with API key and model"""
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

    def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        return_usage: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """Generate content from a single prompt with optional system instruction"""
        try:
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
