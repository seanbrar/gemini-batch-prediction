from typing import Any, List  # noqa: D100, UP035

from .base import BasePromptBuilder


class StructuredPromptBuilder(BasePromptBuilder):
    """Builds a prompt to get a structured JSON object based on a user-provided schema."""  # noqa: E501

    def __init__(self, schema: Any):  # noqa: ANN204, ANN401, D107
        self.schema = schema

    def create_prompt(self, questions: List[str]) -> str:  # noqa: UP006
        """Creates a prompt that instructs the model to populate a JSON object."""
        # Note: The schema itself is passed via the API's `response_schema` config.
        # This prompt provides the instructional context.
        prompt_parts = [
            "Please answer the following questions by generating a single, valid JSON object that "  # noqa: E501
            "strictly conforms to the provided schema.",
            "Do not include markdown formatting, explanations, or any other text outside the final JSON object.",  # noqa: E501
        ]
        prompt_parts.extend(f"Question {i}: {q}" for i, q in enumerate(questions, 1))

        return "\n".join(prompt_parts)
