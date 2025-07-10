"""
Comprehensive unit tests for efficiency and metrics functionality.
"""

from unittest.mock import Mock

from gemini_batch.efficiency.metrics import (
    calculate_cache_savings,
    extract_detailed_usage_metrics,
    extract_usage_metrics,
)


class TestUsageMetricsExtraction:
    """Test usage metrics extraction from API responses."""

    def test_extract_usage_metrics_basic(self):
        """Test extraction of usage metrics from Gemini API response."""
        # Mock API response with usage metadata
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 1000
        mock_response.usage_metadata.candidates_token_count = 500
        mock_response.usage_metadata.cached_content_token_count = 200

        result = extract_usage_metrics(mock_response)

        assert result["prompt_tokens"] == 1000
        assert result["output_tokens"] == 500
        assert result["cached_tokens"] == 200
        assert result["total_tokens"] == 1500  # 1000 + 500
        assert result["effective_prompt_tokens"] == 800  # 1000 - 200
        assert result["cache_hit_ratio"] == 0.2  # 200 / 1000
        assert result["cache_enabled"] is True

    def test_extract_usage_metrics_no_cached_content(self):
        """Test extraction when no content is cached."""
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 800
        mock_response.usage_metadata.candidates_token_count = 400
        mock_response.usage_metadata.cached_content_token_count = 0

        result = extract_usage_metrics(mock_response)

        assert result["prompt_tokens"] == 800
        assert result["output_tokens"] == 400
        assert result["cached_tokens"] == 0
        assert result["total_tokens"] == 1200
        assert result["effective_prompt_tokens"] == 800
        assert result["cache_hit_ratio"] == 0.0
        assert result["cache_enabled"] is False

    def test_extract_usage_metrics_missing_metadata(self):
        """Test extraction when usage metadata is missing."""
        mock_response = Mock()
        mock_response.usage_metadata = None

        result = extract_usage_metrics(mock_response)

        # Should return default values
        assert result["prompt_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["cached_tokens"] == 0
        assert result["total_tokens"] == 0
        assert result["effective_prompt_tokens"] == 0
        assert result["cache_hit_ratio"] == 0.0
        assert result["cache_enabled"] is False

    def test_extract_usage_metrics_partial_metadata(self):
        """Test extraction when some metadata fields are missing."""
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 1000
        # Missing candidates_token_count and cached_content_token_count

        print(
            f"DEBUG: Direct access: {mock_response.usage_metadata.prompt_token_count}"
        )
        print(
            f"DEBUG: getattr: {getattr(mock_response.usage_metadata, 'prompt_token_count', 0)}"
        )

        result = extract_usage_metrics(mock_response)

        print(f"DEBUG: Result: {result}")

        assert result["prompt_tokens"] == 1000
        assert result["output_tokens"] == 0  # Default when missing
        assert result["cached_tokens"] == 0  # Default when missing
        assert result["total_tokens"] == 1000  # prompt_tokens + output_tokens
        assert result["effective_prompt_tokens"] == 1000
        assert result["cache_hit_ratio"] == 0.0
        assert result["cache_enabled"] is False

    def test_extract_usage_metrics_large_numbers(self):
        """Test extraction with large token counts."""
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 50000
        mock_response.usage_metadata.candidates_token_count = 25000
        mock_response.usage_metadata.cached_content_token_count = 10000

        result = extract_usage_metrics(mock_response)

        assert result["prompt_tokens"] == 50000
        assert result["output_tokens"] == 25000
        assert result["cached_tokens"] == 10000
        assert result["total_tokens"] == 75000
        assert result["effective_prompt_tokens"] == 40000
        assert result["cache_hit_ratio"] == 0.2
        assert result["cache_enabled"] is True

    def test_extract_usage_metrics_zero_prompt_tokens(self):
        """Test extraction when prompt tokens is zero (edge case)."""
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 0
        mock_response.usage_metadata.candidates_token_count = 100
        mock_response.usage_metadata.cached_content_token_count = 0

        result = extract_usage_metrics(mock_response)

        assert result["prompt_tokens"] == 0
        assert result["output_tokens"] == 100
        assert result["cached_tokens"] == 0
        assert result["total_tokens"] == 100
        assert result["effective_prompt_tokens"] == 0
        assert result["cache_hit_ratio"] == 0.0  # 0/0 = 0.0
        assert result["cache_enabled"] is False

    def test_extract_usage_metrics_error_handling(self):
        """Test that extraction handles errors gracefully."""
        # Test with completely broken response
        broken_response = Mock()
        broken_response.usage_metadata = "not_a_metadata_object"

        # Should not raise exception
        result = extract_usage_metrics(broken_response)
        assert result["prompt_tokens"] == 0
        assert result["cache_enabled"] is False

    def test_extract_usage_metrics_attribute_error(self):
        """Should handle AttributeError gracefully"""
        mock_response = Mock()
        mock_response.usage_metadata = "not_an_object"

        metrics = extract_usage_metrics(mock_response)

        assert metrics["prompt_tokens"] == 0
        assert metrics["output_tokens"] == 0
        assert metrics["cached_tokens"] == 0
        assert metrics["total_tokens"] == 0
        assert metrics["effective_prompt_tokens"] == 0
        assert metrics["cache_hit_ratio"] == 0.0
        assert metrics["cache_enabled"] is False

    def test_extract_usage_metrics_type_error(self):
        """Should handle TypeError gracefully"""
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        # Make the mock raise TypeError when accessed
        mock_response.usage_metadata.prompt_token_count = Mock(side_effect=TypeError)

        metrics = extract_usage_metrics(mock_response)

        assert metrics["prompt_tokens"] == 0
        assert metrics["output_tokens"] == 0
        assert metrics["cached_tokens"] == 0
        assert metrics["total_tokens"] == 0
        assert metrics["effective_prompt_tokens"] == 0
        assert metrics["cache_hit_ratio"] == 0.0
        assert metrics["cache_enabled"] is False


class TestDetailedUsageMetrics:
    """Test detailed usage metrics extraction."""

    def test_extract_detailed_usage_metrics_with_cache(self):
        """Test detailed extraction with cache analysis."""
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 2000
        mock_response.usage_metadata.candidates_token_count = 1000
        mock_response.usage_metadata.cached_content_token_count = 800

        result = extract_detailed_usage_metrics(
            mock_response, include_cache_analysis=True
        )

        assert result["prompt_tokens"] == 2000
        assert result["output_tokens"] == 1000
        assert result["cached_tokens"] == 800
        assert result["cache_enabled"] is True
        assert "cache_analysis" in result
        assert result["cache_analysis"]["cache_percentage"] == 40.0  # 800/2000 * 100
        assert result["cache_analysis"]["non_cached_tokens"] == 1200
        assert result["cache_analysis"]["cache_efficiency_score"] == 0.4
        assert (
            abs(result["cache_analysis"]["estimated_cost_reduction"] - 0.3) < 0.001
        )  # 0.4 * 0.75

    def test_extract_detailed_usage_metrics_without_cache_analysis(self):
        """Test detailed extraction without cache analysis."""
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 1000
        mock_response.usage_metadata.candidates_token_count = 500
        mock_response.usage_metadata.cached_content_token_count = 0

        result = extract_detailed_usage_metrics(
            mock_response, include_cache_analysis=False
        )

        assert result["prompt_tokens"] == 1000
        assert result["output_tokens"] == 500
        assert result["cached_tokens"] == 0
        assert "cache_analysis" not in result

    def test_extract_detailed_usage_metrics_response_metadata(self):
        """Test that response metadata is included."""
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 100

        result = extract_detailed_usage_metrics(mock_response)

        assert "response_metadata" in result
        assert result["response_metadata"]["has_usage_metadata"] is True
        assert "Mock" in result["response_metadata"]["response_type"]

    def test_extract_detailed_usage_metrics_no_cache_enabled(self):
        """Test detailed extraction when cache is not enabled."""
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 1000
        mock_response.usage_metadata.candidates_token_count = 500
        mock_response.usage_metadata.cached_content_token_count = 0

        result = extract_detailed_usage_metrics(
            mock_response, include_cache_analysis=True
        )

        assert result["cache_enabled"] is False
        assert "cache_analysis" not in result

    def test_extract_detailed_usage_metrics_error_handling(self):
        """Should handle errors gracefully"""
        mock_response = Mock()
        mock_response.usage_metadata = "invalid_object"

        result = extract_detailed_usage_metrics(mock_response)

        assert result["prompt_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["cached_tokens"] == 0
        assert result["cache_enabled"] is False
        assert "response_metadata" in result


class TestCacheSavingsCalculation:
    """Test cache savings calculations."""

    def test_calculate_cache_savings_with_cache(self):
        """Test cache savings calculation when caching is effective."""
        usage_with_cache = {
            "prompt_tokens": 2000,
            "output_tokens": 1000,
            "cached_tokens": 800,
            "total_tokens": 2200,
        }
        usage_without_cache = {
            "prompt_tokens": 2000,
            "output_tokens": 1000,
            "cached_tokens": 0,
            "total_tokens": 3000,
        }

        result = calculate_cache_savings(usage_with_cache, usage_without_cache)

        assert result["tokens_saved"] == 800  # 3000 - 2200
        assert abs(result["cost_savings_ratio"] - 0.267) < 0.001  # 800/3000
        assert result["time_savings_estimate"] == 720.0  # 800 * 0.9
        assert result["cache_effectiveness"] == 0.4  # 800/2000

    def test_calculate_cache_savings_no_cache(self):
        """Test cache savings when no caching occurred."""
        usage_with_cache = {
            "prompt_tokens": 1000,
            "output_tokens": 500,
            "cached_tokens": 0,
            "total_tokens": 1500,
        }
        usage_without_cache = {
            "prompt_tokens": 1000,
            "output_tokens": 500,
            "cached_tokens": 0,
            "total_tokens": 1500,
        }

        result = calculate_cache_savings(usage_with_cache, usage_without_cache)

        assert result["tokens_saved"] == 0
        assert result["cost_savings_ratio"] == 0.0
        assert result["time_savings_estimate"] == 0.0
        assert result["cache_effectiveness"] == 0.0

    def test_calculate_cache_savings_token_reduction(self):
        """Should calculate savings when cache reduces total tokens"""
        usage_with_cache = {
            "prompt_tokens": 80,  # Reduced due to cache
            "output_tokens": 50,
            "cached_tokens": 20,
            "total_tokens": 130,
        }
        usage_without_cache = {
            "prompt_tokens": 100,
            "output_tokens": 50,
            "cached_tokens": 0,
            "total_tokens": 150,
        }

        savings = calculate_cache_savings(usage_with_cache, usage_without_cache)

        assert savings["tokens_saved"] == 20  # 150 - 130
        assert savings["cost_savings_ratio"] == 0.13333333333333333  # 20 / 150
        assert savings["time_savings_estimate"] == 18.0  # 20 * 0.9
        assert savings["cache_effectiveness"] == 0.25  # 20 / 80

    def test_calculate_cache_savings_edge_cases(self):
        """Should handle edge cases gracefully"""
        # Zero tokens without cache
        usage_with_cache = {"cached_tokens": 10, "total_tokens": 50}
        usage_without_cache = {"total_tokens": 0}

        savings = calculate_cache_savings(usage_with_cache, usage_without_cache)

        assert savings["tokens_saved"] == -50  # 0 - 50
        assert savings["cost_savings_ratio"] == -50.0  # -50 / 0 (max 1)
        assert savings["time_savings_estimate"] == 9.0  # 10 * 0.9
        assert savings["cache_effectiveness"] == 10.0  # 10 / 1 (min 1)

        # Zero prompt tokens
        usage_with_cache = {"cached_tokens": 10, "prompt_tokens": 0}
        usage_without_cache = {"total_tokens": 100}

        savings = calculate_cache_savings(usage_with_cache, usage_without_cache)

        assert savings["cache_effectiveness"] == 10.0  # 10 / 1 (min 1)

    def test_calculate_cache_savings_high_cache_effectiveness(self):
        """Test cache savings with very high cache effectiveness."""
        usage_with_cache = {
            "prompt_tokens": 100,
            "output_tokens": 50,
            "cached_tokens": 90,  # 90% cache hit
            "total_tokens": 150,
        }
        usage_without_cache = {
            "prompt_tokens": 100,
            "output_tokens": 50,
            "cached_tokens": 0,
            "total_tokens": 150,
        }

        result = calculate_cache_savings(usage_with_cache, usage_without_cache)

        assert result["cache_effectiveness"] == 0.9  # 90/100
        assert result["time_savings_estimate"] == 81.0  # 90 * 0.9

    def test_calculate_cache_savings_zero_prompt_tokens(self):
        """Test cache savings when prompt tokens is zero."""
        usage_with_cache = {
            "prompt_tokens": 0,
            "output_tokens": 50,
            "cached_tokens": 10,
            "total_tokens": 50,
        }
        usage_without_cache = {
            "prompt_tokens": 0,
            "output_tokens": 50,
            "cached_tokens": 0,
            "total_tokens": 50,
        }

        result = calculate_cache_savings(usage_with_cache, usage_without_cache)

        # When prompt_tokens is 0, cache_effectiveness should be 0
        assert result["cache_effectiveness"] == 0.0
        assert result["tokens_saved"] == 0


class TestMetricsIntegration:
    """Test metrics functionality working together."""

    def test_metrics_workflow_complete(self):
        """Test complete metrics workflow from extraction to analysis."""
        # Create a realistic API response
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 2000
        mock_response.usage_metadata.candidates_token_count = 1000
        mock_response.usage_metadata.cached_content_token_count = 800

        # Extract basic metrics
        basic_metrics = extract_usage_metrics(mock_response)
        assert basic_metrics["cache_enabled"] is True
        assert basic_metrics["cache_hit_ratio"] == 0.4

        # Extract detailed metrics
        detailed_metrics = extract_detailed_usage_metrics(
            mock_response, include_cache_analysis=True
        )
        assert "cache_analysis" in detailed_metrics
        assert detailed_metrics["cache_analysis"]["cache_percentage"] == 40.0

        # Calculate cache savings
        usage_without_cache = {
            "prompt_tokens": 2000,
            "output_tokens": 1000,
            "cached_tokens": 0,
            "total_tokens": 3000,
        }
        savings = calculate_cache_savings(basic_metrics, usage_without_cache)
        assert savings["tokens_saved"] == 800  # 3000 - 2200 (corrected)
        assert savings["cache_effectiveness"] == 0.4

    def test_metrics_error_handling(self):
        """Test that metrics handle errors gracefully throughout the pipeline."""
        # Test with broken response
        broken_response = Mock()
        broken_response.usage_metadata = "invalid"

        # All functions should handle this gracefully
        basic_metrics = extract_usage_metrics(broken_response)
        detailed_metrics = extract_detailed_usage_metrics(broken_response)
        savings = calculate_cache_savings(basic_metrics, basic_metrics)

        assert basic_metrics["prompt_tokens"] == 0
        assert detailed_metrics["prompt_tokens"] == 0
        assert savings["tokens_saved"] == 0

    def test_metrics_consistency(self):
        """Test that different metrics functions return consistent data."""
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 1000
        mock_response.usage_metadata.candidates_token_count = 500
        mock_response.usage_metadata.cached_content_token_count = 200

        basic_metrics = extract_usage_metrics(mock_response)
        detailed_metrics = extract_detailed_usage_metrics(mock_response)

        # Core metrics should be consistent
        assert basic_metrics["prompt_tokens"] == detailed_metrics["prompt_tokens"]
        assert basic_metrics["output_tokens"] == detailed_metrics["output_tokens"]
        assert basic_metrics["cached_tokens"] == detailed_metrics["cached_tokens"]
        assert basic_metrics["cache_enabled"] == detailed_metrics["cache_enabled"]
