"""
Batch processor for text content analysis
"""

from typing import Any, Dict, List, Tuple

from .client import GeminiClient
from .exceptions import APIError, MissingKeyError, NetworkError
from .utils import extract_answers


class BatchProcessor:
    """Process multiple questions efficiently using batch operations"""

    def __init__(self, api_key: str = None):
        """Initialize batch processor"""
        try:
            self.client = GeminiClient(api_key=api_key)
        except MissingKeyError:
            raise
        except NetworkError:
            raise

    def process_batch(
        self, content: str, questions: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Process all questions in a single batch call"""
        try:
            response = self.client.generate_batch(content, questions)

            # Extract individual answers
            answers = extract_answers(response, len(questions))

            return answers

        except (APIError, NetworkError):
            raise

    def process_text_questions(
        self, content: str, questions: List[str]
    ) -> Dict[str, Any]:
        """Process multiple questions about text content"""
        # TODO: Implement batch processing logic
        # TODO: Compare individual vs batch efficiency
        return {
            "question_count": len(questions),
            "answers": [],
            "efficiency": {"efficiency": 1.0},
        }
