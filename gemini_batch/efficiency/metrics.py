"""
Usage metrics extraction from API responses
"""

from typing import Dict


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
