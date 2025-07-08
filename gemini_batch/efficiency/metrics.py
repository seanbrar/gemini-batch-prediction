"""
Usage metrics extraction from API responses
"""

from typing import Any, Dict


def _get_token_count(usage_obj, attr_name: str) -> int:
    """Safely extract token count from usage object"""
    try:
        value = getattr(usage_obj, attr_name, 0)
        # Handle None values and ensure we return an int
        return int(value) if value is not None else 0
    except (ValueError, TypeError):
        # Handle cases where the value can't be converted to int
        return 0


def extract_usage_metrics(response) -> Dict[str, int]:
    """Extract token usage from API response"""
    # Default values
    prompt_tokens = 0
    output_tokens = 0
    cached_tokens = 0

    try:
        usage = response.usage_metadata
        if usage is not None:
            # Use getattr with proper defaults and validation
            prompt_tokens = _get_token_count(usage, "prompt_token_count")
            output_tokens = _get_token_count(usage, "candidates_token_count")
            cached_tokens = _get_token_count(usage, "cached_content_token_count")

    except (AttributeError, TypeError):
        # Handle cases where usage_metadata doesn't exist or isn't accessible
        pass

    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": (prompt_tokens - cached_tokens) + output_tokens,
        "effective_prompt_tokens": prompt_tokens - cached_tokens,
        "cache_hit_ratio": cached_tokens / max(prompt_tokens, 1),
        "cache_enabled": cached_tokens > 0,
    }


def calculate_cache_savings(
    usage_with_cache: Dict[str, int], usage_without_cache: Dict[str, int]
) -> Dict[str, float]:
    """
    Calculate cache savings between cached and non-cached usage.
    """
    cached_tokens = usage_with_cache.get("cached_tokens", 0)
    prompt_tokens = usage_with_cache.get("prompt_tokens", 0)

    if cached_tokens == 0:
        return {
            "tokens_saved": 0,
            "cost_savings_ratio": 0.0,
            "time_savings_estimate": 0.0,
            "cache_effectiveness": 0.0,
        }

    # Calculate token savings
    total_without_cache = usage_without_cache.get("total_tokens", 0)
    total_with_cache = usage_with_cache.get("total_tokens", 0)
    tokens_saved = total_without_cache - total_with_cache

    # Calculate savings ratios
    cost_savings_ratio = tokens_saved / max(total_without_cache, 1)

    # Time savings estimate (cached tokens process ~10x faster)
    time_savings_estimate = cached_tokens * 0.9  # 90% time reduction for cached portion

    # Cache effectiveness (how much of the potential was cached)
    # If prompt_tokens is 0 but cached_tokens > 0, use cached_tokens / 1
    cache_effectiveness = (
        cached_tokens / max(prompt_tokens, 1) if cached_tokens > 0 else 0.0
    )

    return {
        "tokens_saved": tokens_saved,
        "cost_savings_ratio": cost_savings_ratio,
        "time_savings_estimate": time_savings_estimate,
        "cache_effectiveness": cache_effectiveness,
    }


def extract_detailed_usage_metrics(
    response, include_cache_analysis: bool = True
) -> Dict[str, Any]:
    """
    Extract comprehensive usage metrics with optional cache analysis.

    Provides detailed breakdown for efficiency analysis and reporting.
    """
    basic_usage = extract_usage_metrics(response)

    detailed_metrics = {
        **basic_usage,
        "response_metadata": {
            "has_usage_metadata": hasattr(response, "usage_metadata"),
            "response_type": type(response).__name__,
        },
    }

    if include_cache_analysis and basic_usage["cache_enabled"]:
        # Add cache analysis
        cached_tokens = basic_usage["cached_tokens"]
        prompt_tokens = basic_usage["prompt_tokens"]

        detailed_metrics["cache_analysis"] = {
            "cache_percentage": (cached_tokens / max(prompt_tokens, 1)) * 100,
            "non_cached_tokens": prompt_tokens - cached_tokens,
            "cache_efficiency_score": min(cached_tokens / max(prompt_tokens, 1), 1.0),
            "estimated_cost_reduction": (cached_tokens / max(prompt_tokens, 1))
            * 0.75,  # 75% cost reduction for cached tokens
        }

    return detailed_metrics
