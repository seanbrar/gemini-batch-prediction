"""
Answer extraction from batch responses
"""

import re
from typing import List


def extract_answers(
    response_data, question_count: int, is_structured: bool = False
) -> List[str]:
    """Extract individual answers from batch response"""
    if is_structured:
        # Handle structured output response
        if isinstance(response_data, dict) and "parsed" in response_data:
            parsed_data = response_data["parsed"]

            if parsed_data is None:
                # Fallback to text extraction if parsing failed
                response_text = response_data.get("text", "")
                return extract_answers_from_text(response_text, question_count)

            # Convert structured data to answer list
            if isinstance(parsed_data, list):
                # Assuming list of answer objects
                answers = []
                for i, item in enumerate(parsed_data):
                    if i >= question_count:
                        break
                    # Convert to string representation
                    answer_str = (
                        str(item) if item is not None else f"Answer {i + 1}: (No data)"
                    )
                    answers.append(answer_str)

                # Pad with placeholder answers if needed
                while len(answers) < question_count:
                    answers.append(f"Answer {len(answers) + 1}: (No data)")

                return answers[:question_count]
            else:
                # Single structured object - convert to string
                return [str(parsed_data)]
        else:
            # No parsed data available, fallback to text
            response_text = (
                response_data.get("text", "")
                if isinstance(response_data, dict)
                else str(response_data)
            )
            return extract_answers_from_text(response_text, question_count)
    else:
        # Handle text response
        response_text = (
            response_data if isinstance(response_data, str) else str(response_data)
        )
        return extract_answers_from_text(response_text, question_count)


def extract_answers_from_text(response_text: str, question_count: int) -> List[str]:
    """Extract individual answers from text using regex patterns"""
    answers = []

    for i in range(1, question_count + 1):
        # Try multiple extraction patterns for robustness
        # TODO: Improve approach when structured output is not used
        patterns = [
            rf"Answer {i}:\s*(.*?)(?=Answer {i + 1}:|$)",
            rf"{i}\.\s*(.*?)(?={i + 1}\.|$)",
            rf"Question {i}.*?\n\s*(.*?)(?=Question {i + 1}|$)",
        ]
        answer_found = False
        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                answer = match.group(1).strip()
                if answer and len(answer) > 5:
                    answers.append(answer)
                    answer_found = True
                    break

        if not answer_found:
            answers.append(f"Answer {i}: (No answer found)")

    return answers
