"""
Batch processor for text content analysis
"""

import time
from typing import Any, Dict, List, Tuple

from .client import GeminiClient
from .exceptions import APIError, MissingKeyError, NetworkError
from .utils import extract_answers, track_efficiency


class BatchProcessor:
    """Process multiple questions efficiently using batch operations"""

    def __init__(self, api_key: str = None):
        """Initialize batch processor"""
        try:
            self.client = GeminiClient(api_key=api_key)
            self.reset_metrics()
        except MissingKeyError:
            raise
        except NetworkError:
            raise

    def reset_metrics(self):
        """Reset tracking metrics"""
        self.individual_calls = 0
        self.batch_calls = 0
        self.individual_prompt_tokens = 0
        self.individual_output_tokens = 0
        self.batch_prompt_tokens = 0
        self.batch_output_tokens = 0
        self.individual_time = 0.0
        self.batch_time = 0.0

    def process_individual(
        self, content: str, questions: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Process questions individually for comparison"""
        answers = []
        total_prompt_tokens = 0
        total_output_tokens = 0

        start_time = time.time()

        for question in questions:
            try:
                response = self.client.generate_content(
                    prompt=f"Content: {content}\n\nQuestion: {question}\n\nAnswer:",
                    return_usage=True,
                )

                answers.append(response["text"])
                usage = response["usage"]
                total_prompt_tokens += usage["prompt_tokens"]
                total_output_tokens += usage["output_tokens"]
                self.individual_calls += 1

            except (APIError, NetworkError) as e:
                answers.append(f"Error: {str(e)}")

        duration = time.time() - start_time
        self.individual_time += duration
        self.individual_prompt_tokens += total_prompt_tokens
        self.individual_output_tokens += total_output_tokens

        return answers, {
            "calls": self.individual_calls,
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": total_output_tokens,
            "tokens": total_prompt_tokens + total_output_tokens,
            "time": duration,
        }

    def process_batch(
        self, content: str, questions: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Process all questions in a single batch call"""
        start_time = time.time()

        try:
            response = self.client.generate_batch(content, questions, return_usage=True)

            # Extract individual answers
            answers = extract_answers(response["text"], len(questions))

            usage = response["usage"]
            duration = time.time() - start_time

            self.batch_calls += 1
            self.batch_prompt_tokens += usage["prompt_tokens"]
            self.batch_output_tokens += usage["output_tokens"]
            self.batch_time += duration

            return answers, {
                "calls": 1,
                "prompt_tokens": usage["prompt_tokens"],
                "output_tokens": usage["output_tokens"],
                "tokens": usage["total_tokens"],
                "time": duration,
            }

        except (APIError, NetworkError):
            # Fallback to individual processing on batch failure
            return self.process_individual(content, questions)

    def process_text_questions(
        self, content: str, questions: List[str], compare_methods: bool = False
    ) -> Dict[str, Any]:
        """Process multiple questions about text content"""
        if not content or not questions:
            raise ValueError("Content and questions are required")

        self.reset_metrics()

        # Process batch first
        batch_answers, batch_metrics = self.process_batch(content, questions)

        # Conditionally process individually for comparison
        if compare_methods:
            individual_answers, individual_metrics = self.process_individual(
                content, questions
            )
        else:
            individual_answers = None
            individual_metrics = {
                "calls": 0,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "tokens": 0,
                "time": 0.0,
            }

        # Calculate efficiency metrics
        efficiency = track_efficiency(
            individual_calls=individual_metrics["calls"],
            batch_calls=batch_metrics["calls"],
            individual_prompt_tokens=individual_metrics.get("prompt_tokens", 0),
            individual_output_tokens=individual_metrics.get("output_tokens", 0),
            batch_prompt_tokens=batch_metrics.get("prompt_tokens", 0),
            batch_output_tokens=batch_metrics.get("output_tokens", 0),
            individual_time=individual_metrics["time"],
            batch_time=batch_metrics["time"],
        )

        return {
            "question_count": len(questions),
            "batch_answers": batch_answers,
            "efficiency": efficiency,
            "metrics": {
                "batch": batch_metrics,
                "individual": individual_metrics,
            },
            # Conditionally include individual answers with dictionary unpacking
            **({"individual_answers": individual_answers} if compare_methods else {}),
        }
