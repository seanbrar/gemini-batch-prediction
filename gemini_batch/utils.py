"""
Utilities for efficiency tracking and answer processing
"""

import os
import re
from typing import Dict, List, Optional

from .exceptions import MissingKeyError


def parse_env_bool(key: str, default: bool = False) -> bool:
    """Generic environment boolean parser"""
    value = os.getenv(key, "").lower().strip()
    return value in ("true", "1", "yes", "on") if value else default


def get_env_with_fallback(primary_key: str, fallback_key: str) -> Optional[str]:
    """Generic environment variable with fallback"""
    return os.getenv(primary_key) or os.getenv(fallback_key)


def validate_api_key_format(api_key: str) -> bool:
    """Generic API key format validation - could be used anywhere"""
    if not api_key or not isinstance(api_key, str):
        raise MissingKeyError("API key must be a non-empty string")

    # Basic length check
    api_key = api_key.strip()
    if len(api_key) < 30:  # Too short
        raise MissingKeyError("API key appears to be invalid (too short)")

    return True


def _calculate_token_efficiency(prompt_tokens: int, output_tokens: int) -> float:
    """Calculate token efficiency: output tokens / total tokens"""
    total_tokens = prompt_tokens + output_tokens
    return output_tokens / max(total_tokens, 1)


def _calculate_efficiency_metrics(
    individual_prompt_tokens: int,
    individual_output_tokens: int,
    batch_prompt_tokens: int,
    batch_output_tokens: int,
    individual_time: float,
    batch_time: float,
) -> Dict[str, float]:
    """Calculate all efficiency metrics when comparison data is available"""

    # Calculate token efficiency for each approach (generated/total ratio)
    individual_token_efficiency = _calculate_token_efficiency(
        individual_prompt_tokens, individual_output_tokens
    )
    batch_token_efficiency = _calculate_token_efficiency(
        batch_prompt_tokens, batch_output_tokens
    )

    # Total tokens used by each approach
    individual_total = individual_prompt_tokens + individual_output_tokens
    batch_total = batch_prompt_tokens + batch_output_tokens

    # Key metric: How many times fewer tokens does batch use?
    token_savings_ratio = individual_total / max(batch_total, 1)

    time_efficiency = individual_time / max(batch_time, 1) if batch_time > 0 else 1.0

    # Overall efficiency is the token savings ratio
    overall_efficiency = token_savings_ratio

    # Target: 3x minimum efficiency improvement
    meets_target = token_savings_ratio >= 3.0

    return {
        "individual_token_efficiency": individual_token_efficiency,
        "batch_token_efficiency": batch_token_efficiency,
        "token_efficiency_ratio": token_savings_ratio,
        "time_efficiency": time_efficiency,
        "overall_efficiency": overall_efficiency,
        "meets_target": meets_target,
        "individual_total_tokens": individual_total,
        "batch_total_tokens": batch_total,
    }


def track_efficiency(
    individual_calls: int,
    batch_calls: int,
    individual_prompt_tokens: int = 0,
    individual_output_tokens: int = 0,
    batch_prompt_tokens: int = 0,
    batch_output_tokens: int = 0,
    individual_time: float = 0.0,
    batch_time: float = 0.0,
) -> Dict[str, float]:
    """Calculate efficiency metrics for batch processing"""

    # Determine if comparison data is available
    comparison_available = individual_calls > 0 and batch_calls > 0

    if (
        comparison_available
        and individual_prompt_tokens > 0
        and batch_prompt_tokens > 0
    ):
        # Full comparison available
        metrics = _calculate_efficiency_metrics(
            individual_prompt_tokens,
            individual_output_tokens,
            batch_prompt_tokens,
            batch_output_tokens,
            individual_time,
            batch_time,
        )

    else:
        # Limited data - calculate what we can
        batch_efficiency = None
        individual_efficiency = None

        if batch_prompt_tokens > 0 and batch_output_tokens > 0:
            batch_efficiency = _calculate_token_efficiency(
                batch_prompt_tokens, batch_output_tokens
            )

        if individual_prompt_tokens > 0 and individual_output_tokens > 0:
            individual_efficiency = _calculate_token_efficiency(
                individual_prompt_tokens, individual_output_tokens
            )

        metrics = {
            "individual_token_efficiency": individual_efficiency
            if comparison_available
            else None,
            "batch_token_efficiency": batch_efficiency
            if comparison_available
            else None,
            "token_efficiency_ratio": None,
            "time_efficiency": None,
            "overall_efficiency": None,
            "meets_target": batch_calls > 0,  # Success if batch processing worked
        }

    # Add comparison flag to all results
    metrics["comparison_available"] = comparison_available
    return metrics


def extract_usage_metrics(response) -> Dict[str, int]:
    """Extract token usage from API response"""
    try:
        usage = response.usage_metadata
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0

        return {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "total_tokens": prompt_tokens + output_tokens,
        }
    except (AttributeError, TypeError):
        return {
            "prompt_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }


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


def calculate_quality_score(
    individual_answers: List[str], batch_answers: List[str]
) -> float:
    """Calculate quality comparison between individual and batch answers"""
    # Handle case where individual answers aren't available
    if not individual_answers or not batch_answers:
        return None

    if len(individual_answers) != len(batch_answers):
        return 0.0

    quality_scores = []

    for ind, batch in zip(individual_answers, batch_answers):
        # Completeness check (both answers should be substantive)
        ind_complete = len(ind.strip()) > 10
        batch_complete = len(batch.strip()) > 10
        completeness = 1.0 if (ind_complete and batch_complete) else 0.5

        # Word overlap similarity
        ind_words = set(ind.lower().split())
        batch_words = set(batch.lower().split())

        if len(ind_words.union(batch_words)) > 0:
            overlap = len(ind_words.intersection(batch_words)) / len(
                ind_words.union(batch_words)
            )
        else:
            overlap = 0.0

        # Length similarity (with a preference for similar-length answers)
        length_ratio = min(len(ind), len(batch)) / max(len(ind), len(batch), 1)

        # Combined score
        score = (completeness * 0.5) + (overlap * 0.3) + (length_ratio * 0.2)
        quality_scores.append(score)

    return sum(quality_scores) / len(quality_scores)


# Backward compatibility alias
validate_api_key = validate_api_key_format
