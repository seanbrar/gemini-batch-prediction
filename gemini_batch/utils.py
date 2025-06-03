"""
Utilities for efficiency tracking and answer processing
"""

from typing import Dict, List

from .exceptions import MissingKeyError


def track_efficiency(individual_calls: int, batch_calls: int) -> Dict[str, float]:
    """Calculate efficiency metrics for batch processing"""
    # TODO: Add more sophisticated metrics
    if batch_calls == 0:
        return {"efficiency": 0}

    return {"efficiency": individual_calls / batch_calls}


def extract_answers(response_text: str, question_count: int) -> List[str]:
    """Extract individual answers from batch response"""
    # TODO: Implement robust answer extraction
    # Placeholder: simple split
    return [f"Answer {i + 1}" for i in range(question_count)]


def validate_api_key(api_key: str) -> bool:
    """Basic API key format validation"""
    if not api_key or not isinstance(api_key, str):
        raise MissingKeyError("API key must be a non-empty string")

    # Basic length check
    api_key = api_key.strip()
    if len(api_key) < 30:  # Too short
        raise MissingKeyError("API key appears to be invalid (too short)")

    return True
