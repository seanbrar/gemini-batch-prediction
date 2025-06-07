"""
Gemini API client for batch processing experiments
"""

import os
from typing import List, Optional

from google import genai
from google.genai import types

from .exceptions import APIError, MissingKeyError, NetworkError


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
        self, prompt: str, system_instruction: Optional[str] = None
    ) -> str:
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
            return response.text
        except (ConnectionError, TimeoutError) as e:
            raise NetworkError(f"Network connection failed: {e}") from e
        except Exception as e:
            raise APIError(f"API call failed: {e}") from e

    def generate_batch(self, content: str, questions: List[str]) -> str:
        """Generate batch response for multiple questions about content"""
        # TODO: Implement optimized batch processing
        # For now, simple concatenation
        batch_prompt = f"Content: {content}\n\nQuestions: {', '.join(questions)}"
        return self.generate_content(batch_prompt)
