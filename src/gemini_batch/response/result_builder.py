import logging  # noqa: D100
from typing import Any, Dict, List, Optional  # noqa: UP035

from .types import ProcessingMetrics, ProcessingOptions  # noqa: TC001

log = logging.getLogger(__name__)


class ResultBuilder:
    """Unified result building for different processing modes"""  # noqa: D415

    def __init__(self, efficiency_calculator):  # noqa: ANN001, ANN204, D107
        self.efficiency_calculator = efficiency_calculator

    def build_standard_result(  # noqa: PLR0913
        self,
        questions: List[str],  # noqa: UP006
        batch_answers: List[str],  # noqa: UP006
        batch_metrics: "ProcessingMetrics",
        individual_metrics: "ProcessingMetrics",
        individual_answers: Optional[List[str]],  # noqa: UP006, UP045
        config: "ProcessingOptions",
    ) -> Dict[str, Any]:  # noqa: UP006
        """Build result for standard processing mode"""  # noqa: D415
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

        # Add processing time from the primary metrics
        result["processing_time"] = batch_metrics.time

        # Add cache summary if caching was used
        if batch_metrics.cached_tokens > 0 or individual_metrics.cached_tokens > 0:
            result["cache_summary"] = {
                "cache_enabled": True,
                "batch_cache_hit_ratio": batch_metrics.cache_hit_ratio,
                "individual_cache_hit_ratio": individual_metrics.cache_hit_ratio,
                "tokens_saved": individual_metrics.total_tokens
                - batch_metrics.total_tokens,
                "cache_cost_benefit": efficiency.get("cache_efficiency", {}).get(
                    "cache_improvement_factor", 1.0  # noqa: COM812
                ),
            }
        else:
            result["cache_summary"] = {"cache_enabled": False}

        if efficiency.get("comparison_available"):
            log.info(
                "Batch processing completed successfully. Token efficiency: %.2fx",
                efficiency["token_efficiency_ratio"],
            )
        else:
            log.info("Batch processing completed successfully.")

        self._add_optional_data(result, batch_metrics, individual_answers, config)
        return result

    def enhance_response_processor_result(
        self,
        result: Dict[str, Any],  # noqa: UP006
        batch_metrics: "ProcessingMetrics",
        individual_metrics: "ProcessingMetrics",
        individual_answers: List[str],  # noqa: UP006
    ) -> None:
        """Add efficiency comparison metrics to ResponseProcessor result"""  # noqa: D415
        efficiency = self.efficiency_calculator(individual_metrics, batch_metrics)

        result.update(
            {
                "efficiency": efficiency,
                "metrics": {
                    "batch": batch_metrics.to_dict(),
                    "individual": individual_metrics.to_dict(),
                },
                "individual_answers": individual_answers,
            }  # noqa: COM812
        )

    def _add_optional_data(
        self,
        result: Dict[str, Any],  # noqa: UP006
        batch_metrics: "ProcessingMetrics",
        individual_answers: Optional[List[str]],  # noqa: UP006, UP045
        config: "ProcessingOptions",
    ) -> None:
        """Add optional data to result dictionary"""  # noqa: D415
        # Add structured data if available
        if batch_metrics.structured_output and batch_metrics.structured_output.get(
            "structured_data"  # noqa: COM812
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
