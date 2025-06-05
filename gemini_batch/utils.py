"""
Utilities for efficiency tracking and answer processing
"""

import re
from typing import Dict, List

from .exceptions import MissingKeyError


def track_efficiency(individual_calls: int, batch_calls: int) -> Dict[str, float]:
    """Calculate efficiency metrics for batch processing"""
    # TODO: Add more sophisticated metrics
    if batch_calls == 0:
        return {"efficiency": 0}

    return {"efficiency": individual_calls / batch_calls}


def extract_answers(response_text: str, question_count: int) -> List[str]:
    """Extract individual answers from batch response text"""
    answers = []

    for i in range(1, question_count + 1):
        # Try multiple extraction patterns for robustness
        # TODO: Optimize through system instructions and structured output
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


def validate_api_key(api_key: str) -> bool:
    """Basic API key format validation"""
    if not api_key or not isinstance(api_key, str):
        raise MissingKeyError("API key must be a non-empty string")

    # Basic length check
    api_key = api_key.strip()
    if len(api_key) < 30:  # Too short
        raise MissingKeyError("API key appears to be invalid (too short)")

    return True
