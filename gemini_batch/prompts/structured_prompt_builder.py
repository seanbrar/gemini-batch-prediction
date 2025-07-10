from typing import Any, List

from .base import BasePromptBuilder


class StructuredPromptBuilder(BasePromptBuilder):
    """Builds a prompt to get a structured JSON object based on a user-provided schema."""

    def __init__(self, schema: Any):
        self.schema = schema

    def create_prompt(self, questions: List[str]) -> str:
        """Creates a prompt that instructs the model to populate a JSON object."""
        # Note: The schema itself is passed via the API's `response_schema` config.
        # This prompt provides the instructional context.
        prompt_parts = [
            "Please answer the following questions by generating a single, valid JSON object that "
            "strictly conforms to the provided schema.",
            "Do not include markdown formatting, explanations, or any other text outside the final JSON object.",
        ]
        prompt_parts.extend(f"Question {i}: {q}" for i, q in enumerate(questions, 1))

        return "\n".join(prompt_parts)
