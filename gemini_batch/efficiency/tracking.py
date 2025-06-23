"""
Efficiency tracking and metrics calculation for batch processing
"""

from typing import Dict

from ..constants import TARGET_EFFICIENCY_RATIO


def calculate_token_efficiency(prompt_tokens: int, output_tokens: int) -> float:
    """Calculate token efficiency: output tokens / total tokens"""
    total_tokens = prompt_tokens + output_tokens
    return output_tokens / max(total_tokens, 1)


def calculate_efficiency_metrics(
    individual_prompt_tokens: int,
    individual_output_tokens: int,
    batch_prompt_tokens: int,
    batch_output_tokens: int,
    individual_time: float,
    batch_time: float,
) -> Dict[str, float]:
    """Calculate all efficiency metrics when comparison data is available"""

    # Calculate token efficiency for each approach (generated/total ratio)
    individual_token_efficiency = calculate_token_efficiency(
        individual_prompt_tokens, individual_output_tokens
    )
    batch_token_efficiency = calculate_token_efficiency(
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
    # (with 1% relative tolerance for floating-point precision)
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
        metrics = calculate_efficiency_metrics(
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
            batch_efficiency = calculate_token_efficiency(
                batch_prompt_tokens, batch_output_tokens
            )

        if individual_prompt_tokens > 0 and individual_output_tokens > 0:
            individual_efficiency = calculate_token_efficiency(
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
