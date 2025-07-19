"""
Efficiency tracking and metrics calculation for batch processing
"""  # noqa: D200, D212, D415

from typing import Dict  # noqa: UP035

from ..constants import TARGET_EFFICIENCY_RATIO  # noqa: TID252


def calculate_token_efficiency(prompt_tokens: int, output_tokens: int) -> float:
    """Calculate token efficiency: output tokens / total tokens"""  # noqa: D415
    total_tokens = prompt_tokens + output_tokens
    return output_tokens / max(total_tokens, 1)


def calculate_cache_aware_efficiency_metrics(  # noqa: PLR0913
    individual_prompt_tokens: int,
    individual_output_tokens: int,
    individual_cached_tokens: int,
    batch_prompt_tokens: int,
    batch_output_tokens: int,
    batch_cached_tokens: int,
    individual_time: float,
    batch_time: float,
) -> Dict[str, float]:  # noqa: UP006
    """
    Calculate efficiency metrics.

    Accounts for cached tokens in efficiency calculations to provide
    more accurate cost and performance comparisons.
    """  # noqa: D202, D212

    # Calculate effective tokens (excluding cached portions for cost estimation)
    individual_effective_prompt = individual_prompt_tokens - individual_cached_tokens
    batch_effective_prompt = batch_prompt_tokens - batch_cached_tokens

    individual_token_efficiency = calculate_token_efficiency(
        individual_prompt_tokens, individual_output_tokens  # noqa: COM812
    )
    batch_token_efficiency = calculate_token_efficiency(
        batch_prompt_tokens, batch_output_tokens  # noqa: COM812
    )

    # Cache-aware efficiency (based on effective tokens)
    individual_effective_total = individual_effective_prompt + individual_output_tokens
    batch_effective_total = batch_effective_prompt + batch_output_tokens

    # Total tokens used by each approach
    individual_total = individual_prompt_tokens + individual_output_tokens
    batch_total = batch_prompt_tokens + batch_output_tokens

    # Key metric: How many times fewer tokens does batch use?
    token_savings_ratio = individual_total / max(batch_total, 1)

    # Cache-aware cost efficiency (based on effective cost)
    cost_efficiency_ratio = individual_effective_total / max(batch_effective_total, 1)

    # Time efficiency
    time_efficiency = individual_time / max(batch_time, 1) if batch_time > 0 else 1.0

    # Cache efficiency metrics
    cache_metrics = _calculate_cache_efficiency_metrics(
        individual_cached_tokens,
        batch_cached_tokens,
        individual_prompt_tokens,
        batch_prompt_tokens,
    )

    # Overall efficiency (weighted combination of token and cache efficiency)
    overall_efficiency = max(token_savings_ratio, cost_efficiency_ratio)

    # Target: 3x minimum efficiency improvement
    meets_target = token_savings_ratio >= (TARGET_EFFICIENCY_RATIO * 0.99)

    return {
        "individual_token_efficiency": individual_token_efficiency,
        "batch_token_efficiency": batch_token_efficiency,
        "token_efficiency_ratio": token_savings_ratio,
        "time_efficiency": time_efficiency,
        "overall_efficiency": overall_efficiency,
        "meets_target": meets_target,
        "individual_total_tokens": individual_total,
        "batch_total_tokens": batch_total,
        # Cache-aware metrics
        "cost_efficiency_ratio": cost_efficiency_ratio,
        "individual_effective_tokens": individual_effective_total,
        "batch_effective_tokens": batch_effective_total,
        "cache_efficiency": cache_metrics,
    }


def _calculate_cache_efficiency_metrics(
    individual_cached_tokens: int,
    batch_cached_tokens: int,
    individual_prompt_tokens: int,
    batch_prompt_tokens: int,
) -> Dict[str, float]:  # noqa: UP006
    """Calculate cache-specific efficiency metrics"""  # noqa: D202, D415

    individual_cache_ratio = individual_cached_tokens / max(individual_prompt_tokens, 1)
    batch_cache_ratio = batch_cached_tokens / max(batch_prompt_tokens, 1)

    # Cache efficiency improvement
    cache_improvement = batch_cache_ratio / max(individual_cache_ratio, 0.01)

    # Cache utilization score
    cache_utilization = (individual_cached_tokens + batch_cached_tokens) / max(
        individual_prompt_tokens + batch_prompt_tokens, 1  # noqa: COM812
    )

    return {
        "individual_cache_ratio": individual_cache_ratio,
        "batch_cache_ratio": batch_cache_ratio,
        "cache_improvement_factor": cache_improvement,
        "overall_cache_utilization": cache_utilization,
        "cache_enabled": batch_cached_tokens > 0 or individual_cached_tokens > 0,
    }


def track_efficiency(  # noqa: PLR0913
    individual_calls: int,
    batch_calls: int,
    individual_prompt_tokens: int = 0,
    individual_output_tokens: int = 0,
    batch_prompt_tokens: int = 0,
    batch_output_tokens: int = 0,
    individual_time: float = 0.0,
    batch_time: float = 0.0,
    # Cache-aware parameters
    individual_cached_tokens: int = 0,
    batch_cached_tokens: int = 0,
    include_cache_metrics: bool = True,  # noqa: FBT001, FBT002
) -> Dict[str, float]:  # noqa: UP006
    """Calculate efficiency metrics for batch processing with optional cache awareness"""  # noqa: D202, D415, E501

    # Determine if comparison data is available
    comparison_available = individual_calls > 0 and batch_calls > 0

    if (
        comparison_available
        and individual_prompt_tokens > 0
        and batch_prompt_tokens > 0
    ):
        # Check if we should use cache-aware calculation
        has_cache_data = include_cache_metrics and (
            individual_cached_tokens > 0 or batch_cached_tokens > 0
        )

        if has_cache_data:
            # Cache-aware efficiency calculation
            metrics = calculate_cache_aware_efficiency_metrics(
                individual_prompt_tokens,
                individual_output_tokens,
                individual_cached_tokens,
                batch_prompt_tokens,
                batch_output_tokens,
                batch_cached_tokens,
                individual_time,
                batch_time,
            )
        else:
            # Traditional efficiency calculation (without cache awareness)
            # Calculate token efficiency for each approach
            individual_token_efficiency = calculate_token_efficiency(
                individual_prompt_tokens, individual_output_tokens  # noqa: COM812
            )
            batch_token_efficiency = calculate_token_efficiency(
                batch_prompt_tokens, batch_output_tokens  # noqa: COM812
            )

            # Total tokens used by each approach
            individual_total = individual_prompt_tokens + individual_output_tokens
            batch_total = batch_prompt_tokens + batch_output_tokens

            # Key metric: How many times fewer tokens does batch use?
            token_savings_ratio = individual_total / max(batch_total, 1)
            time_efficiency = (
                individual_time / max(batch_time, 1) if batch_time > 0 else 1.0
            )

            metrics = {
                "individual_token_efficiency": individual_token_efficiency,
                "batch_token_efficiency": batch_token_efficiency,
                "token_efficiency_ratio": token_savings_ratio,
                "time_efficiency": time_efficiency,
                "overall_efficiency": token_savings_ratio,
                "meets_target": token_savings_ratio >= (TARGET_EFFICIENCY_RATIO * 0.99),
                "individual_total_tokens": individual_total,
                "batch_total_tokens": batch_total,
            }

    else:
        # Limited data - calculate what we can
        batch_efficiency = None
        individual_efficiency = None

        if batch_prompt_tokens > 0 and batch_output_tokens > 0:
            batch_efficiency = calculate_token_efficiency(
                batch_prompt_tokens, batch_output_tokens  # noqa: COM812
            )

        if individual_prompt_tokens > 0 and individual_output_tokens > 0:
            individual_efficiency = calculate_token_efficiency(
                individual_prompt_tokens, individual_output_tokens  # noqa: COM812
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

        # Add empty cache metrics for consistency
        if include_cache_metrics:
            metrics["cache_efficiency"] = {
                "individual_cache_ratio": individual_cached_tokens
                / max(individual_prompt_tokens, 1),
                "batch_cache_ratio": batch_cached_tokens / max(batch_prompt_tokens, 1),
                "cache_enabled": individual_cached_tokens > 0
                or batch_cached_tokens > 0,
            }

    # Add comparison flag to all results
    metrics["comparison_available"] = comparison_available

    # Add cache-awareness flag
    metrics["cache_aware_calculation"] = include_cache_metrics and (
        individual_cached_tokens > 0 or batch_cached_tokens > 0
    )

    return metrics


def analyze_cache_efficiency_impact(
    traditional_metrics: Dict[str, float], cache_aware_metrics: Dict[str, float]  # noqa: COM812, UP006
) -> Dict[str, float]:  # noqa: UP006
    """
    Analyze the impact of cache efficiency on overall performance.

    Compares traditional efficiency calculations with cache-aware ones
    to quantify the benefit of caching.
    """  # noqa: D202, D212

    traditional_efficiency = traditional_metrics.get("token_efficiency_ratio", 1.0)
    cache_aware_efficiency = cache_aware_metrics.get("cost_efficiency_ratio", 1.0)

    cache_benefit = cache_aware_efficiency / max(traditional_efficiency, 0.01)

    cache_metrics = cache_aware_metrics.get("cache_efficiency", {})

    return {
        "cache_amplification_factor": cache_benefit,
        "additional_efficiency_from_cache": cache_benefit - 1.0,
        "cache_utilization_score": cache_metrics.get("overall_cache_utilization", 0.0),
        "cache_effectiveness": cache_metrics.get("cache_improvement_factor", 1.0),
        "recommended_caching": cache_benefit > 1.1,  # 10% improvement threshold  # noqa: PLR2004
    }
