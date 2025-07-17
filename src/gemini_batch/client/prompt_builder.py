"""
Prompt construction for single and batch content generation
"""

from typing import Any, List, Optional


class PromptBuilder:
    """Builds prompts for single and batch content generation"""

    def create_batch_prompt(
        self, questions: List[str], response_schema: Optional[Any] = None
    ) -> str:
        """Create batch prompt for multiple questions with optional structured output"""
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
        """Pass through single prompt unchanged"""
        return prompt
