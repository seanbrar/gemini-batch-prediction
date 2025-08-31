from .base import BasePromptBuilder  # noqa: D100


class BatchPromptBuilder(BasePromptBuilder):
    """Builds a prompt to get unstructured answers in a structured JSON list."""

    def create_prompt(self, questions: list[str]) -> str:
        """Creates a prompt that instructs the model to return a JSON array of strings."""
        prompt_parts = ["Please answer each of the following questions."]
        prompt_parts.extend(f"Question {i}: {q}" for i, q in enumerate(questions, 1))

        instruction = (
            "\nYour output MUST be a single, valid JSON array of strings, where each "
            "string is the answer to a question in the corresponding order. "
            "Do not include any other text, formatting, or explanations outside of the JSON array."
        )
        prompt_parts.append(instruction)

        return "\n".join(prompt_parts)
