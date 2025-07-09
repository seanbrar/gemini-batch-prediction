from typing import Any, Dict, List, Optional

from .types import ProcessingMetrics, ProcessingOptions


class ResultBuilder:
    """Unified result building for different processing modes"""

    def __init__(self, efficiency_calculator):
        self.efficiency_calculator = efficiency_calculator

    def build_standard_result(
        self,
        questions: List[str],
        batch_answers: List[str],
        batch_metrics: "ProcessingMetrics",
        individual_metrics: "ProcessingMetrics",
        individual_answers: Optional[List[str]],
        config: "ProcessingOptions",
    ) -> Dict[str, Any]:
        """Build result for standard processing mode"""
        # Check if batch processing failed (0 calls) and we have individual metrics
        if batch_metrics.calls == 0 and individual_metrics.calls > 0:
            # Batch failed, use individual metrics as primary
            efficiency = self.efficiency_calculator(individual_metrics, batch_metrics)

            result = {
                "question_count": len(questions),
                "answers": batch_answers,  # These are actually individual answers
                "efficiency": efficiency,
                "metrics": {
                    "batch": batch_metrics.to_dict(),
                    "individual": individual_metrics.to_dict(),
                },
            }
        else:
            # Normal batch processing
            efficiency = self.efficiency_calculator(individual_metrics, batch_metrics)

            result = {
                "question_count": len(questions),
                "answers": batch_answers,
                "efficiency": efficiency,
                "metrics": {
                    "batch": batch_metrics.to_dict(),
                    "individual": individual_metrics.to_dict(),
                },
            }

        # Add cache summary if caching was used
        if batch_metrics.cached_tokens > 0 or individual_metrics.cached_tokens > 0:
            result["cache_summary"] = {
                "cache_enabled": True,
                "batch_cache_hit_ratio": batch_metrics.cache_hit_ratio,
                "individual_cache_hit_ratio": individual_metrics.cache_hit_ratio,
                "tokens_saved": individual_metrics.total_tokens
                - batch_metrics.total_tokens,
                "cache_cost_benefit": efficiency.get("cache_efficiency", {}).get(
                    "cache_improvement_factor", 1.0
                ),
            }
        else:
            result["cache_summary"] = {"cache_enabled": False}

        self._add_optional_data(result, batch_metrics, individual_answers, config)
        return result

    def enhance_response_processor_result(
        self,
        result: Dict[str, Any],
        batch_metrics: "ProcessingMetrics",
        individual_metrics: "ProcessingMetrics",
        individual_answers: List[str],
    ) -> None:
        """Add efficiency comparison metrics to ResponseProcessor result"""
        efficiency = self.efficiency_calculator(individual_metrics, batch_metrics)

        result.update(
            {
                "efficiency": efficiency,
                "metrics": {
                    "batch": batch_metrics.to_dict(),
                    "individual": individual_metrics.to_dict(),
                },
                "individual_answers": individual_answers,
            }
        )

    def _add_optional_data(
        self,
        result: Dict[str, Any],
        batch_metrics: "ProcessingMetrics",
        individual_answers: Optional[List[str]],
        config: "ProcessingOptions",
    ) -> None:
        """Add optional data to result dictionary"""
        # Add structured data if available
        if batch_metrics.structured_output and batch_metrics.structured_output.get(
            "structured_data"
        ):
            result["structured_data"] = batch_metrics.structured_output[
                "structured_data"
            ]

        # Add individual answers if comparison was requested
        if config.compare_methods and individual_answers:
            result["individual_answers"] = individual_answers

        # Add usage information if requested
        if config.return_usage:
            result["usage"] = {
                "prompt_tokens": batch_metrics.prompt_tokens,
                "output_tokens": batch_metrics.output_tokens,
                "total_tokens": batch_metrics.total_tokens,
                "cached_tokens": batch_metrics.cached_tokens,
                "effective_tokens": batch_metrics.effective_tokens,
                "cache_hit_ratio": batch_metrics.cache_hit_ratio,
            }
