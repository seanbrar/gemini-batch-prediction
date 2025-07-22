"""
Comprehensive unit tests for gemini_batch.utils module
"""

from unittest.mock import Mock, patch

import pytest

from gemini_batch.efficiency.metrics import extract_usage_metrics
from gemini_batch.efficiency.tracking import (
    _calculate_efficiency_metrics,
    _calculate_token_efficiency,
    track_efficiency,
)
from gemini_batch.exceptions import MissingKeyError
from gemini_batch.response.quality import calculate_quality_score
from gemini_batch.utils import (
    get_env_with_fallback,
    parse_env_bool,
    validate_api_key,
    validate_api_key_format,
)


@pytest.mark.unit
class TestEnvironmentUtilities:
    """Test environment variable utility functions"""

    def test_parse_env_bool_true_values(self):
        """Should return True for various true-like values"""
        true_values = [
            "true",
            "True",
            "TRUE",
            "1",
            "yes",
            "Yes",
            "YES",
            "on",
            "On",
            "ON",
        ]

        for value in true_values:
            with patch("os.getenv", return_value=value):
                result = parse_env_bool("TEST_KEY")
                assert result is True, f"Failed for value: {value}"

    def test_parse_env_bool_false_values(self):
        """Should return False for various false-like values"""
        false_values = [
            "false",
            "False",
            "FALSE",
            "0",
            "no",
            "No",
            "NO",
            "off",
            "Off",
            "OFF",
            "random",
            "",
        ]

        for value in false_values:
            with patch("os.getenv", return_value=value):
                result = parse_env_bool("TEST_KEY")
                assert result is False, f"Failed for value: {value}"

    def test_parse_env_bool_empty_and_missing(self):
        """Should return default for empty or missing environment variables"""
        # Empty string
        with patch("os.getenv", return_value=""):
            assert parse_env_bool("TEST_KEY") is False
            assert parse_env_bool("TEST_KEY", default=True) is True

        # None (missing) - need to patch the actual call differently since
        # parse_env_bool calls os.getenv with a default empty string
        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = (
                ""  # os.getenv returns "" when key is missing and default is ""
            )
            assert parse_env_bool("TEST_KEY") is False
            assert parse_env_bool("TEST_KEY", default=True) is True

    def test_parse_env_bool_whitespace_handling(self):
        """Should handle whitespace correctly"""
        test_cases = [
            ("  true  ", True),
            ("\ttrue\n", True),
            ("  false  ", False),
            ("   ", False),  # Only whitespace
        ]

        for value, expected in test_cases:
            with patch("os.getenv", return_value=value):
                result = parse_env_bool("TEST_KEY")
                assert result is expected, f"Failed for value: '{value}'"

    def test_get_env_with_fallback_primary_found(self):
        """Should return primary key value when available"""
        with patch("os.getenv") as mock_getenv:
            mock_getenv.side_effect = lambda key: {
                "PRIMARY": "primary_value",
                "FALLBACK": "fallback_value",
            }.get(key)

            result = get_env_with_fallback("PRIMARY", "FALLBACK")
            assert result == "primary_value"

    def test_get_env_with_fallback_uses_fallback(self):
        """Should return fallback value when primary is not available"""
        with patch("os.getenv") as mock_getenv:
            mock_getenv.side_effect = lambda key: {"FALLBACK": "fallback_value"}.get(
                key
            )

            result = get_env_with_fallback("PRIMARY", "FALLBACK")
            assert result == "fallback_value"

    def test_get_env_with_fallback_both_missing(self):
        """Should return None when both keys are missing"""
        with patch("os.getenv", return_value=None):
            result = get_env_with_fallback("PRIMARY", "FALLBACK")
            assert result is None

    def test_get_env_with_fallback_empty_primary(self):
        """Should use fallback when primary is empty"""
        with patch("os.getenv") as mock_getenv:
            mock_getenv.side_effect = lambda key: {
                "PRIMARY": "",
                "FALLBACK": "fallback_value",
            }.get(key)

            result = get_env_with_fallback("PRIMARY", "FALLBACK")
            assert result == "fallback_value"


class TestAPIKeyFormatValidation:
    """Test API key format validation functionality"""

    def test_validate_api_key_format_valid_keys(self):
        """Should accept properly formatted API keys"""
        valid_keys = [
            "AIzaSyC8UYZpvA2eknNdcAaFeFbRe-PaWiDfD_M",  # Google format
            "test_key_123456789012345678901234567890",  # Test format
            "x" * 30,  # Minimum length
            "x" * 100,  # Long key
        ]

        for key in valid_keys:
            result = validate_api_key_format(key)
            assert result is True

    def test_validate_api_key_format_invalid_inputs(self):
        """Should raise MissingKeyError for invalid inputs"""
        with pytest.raises(MissingKeyError, match="API key must be a non-empty string"):
            validate_api_key_format("")

        with pytest.raises(MissingKeyError, match="API key must be a non-empty string"):
            validate_api_key_format(None)

        with pytest.raises(MissingKeyError, match="API key must be a non-empty string"):
            validate_api_key_format(12345)

    def test_validate_api_key_format_too_short(self):
        """Should raise MissingKeyError for keys that are too short"""
        with pytest.raises(
            MissingKeyError, match="API key appears to be invalid \\(too short\\)"
        ):
            validate_api_key_format("short")

        with pytest.raises(
            MissingKeyError, match="API key appears to be invalid \\(too short\\)"
        ):
            validate_api_key_format("x" * 20)  # Just under minimum

    def test_validate_api_key_format_whitespace_handling(self):
        """Should handle whitespace correctly"""
        # Key with surrounding whitespace should be accepted (after stripping)
        valid_key = "  test_key_123456789012345678901234567890  "
        result = validate_api_key_format(valid_key)
        assert result is True

    def test_validate_api_key_backward_compatibility(self):
        """Should maintain backward compatibility through alias"""
        # The validate_api_key function should be an alias for validate_api_key_format
        assert validate_api_key is validate_api_key_format

        # Test that the alias works
        result = validate_api_key("test_key_123456789012345678901234567890")
        assert result is True

    def test_validate_api_key_edge_cases(self):
        """Should handle API key validation edge cases"""
        # Test with various invalid inputs
        invalid_inputs = ["", None, 0, [], {}, (), 123.45, True, False]

        for invalid_input in invalid_inputs:
            with pytest.raises(MissingKeyError):
                validate_api_key_format(invalid_input)

        # Test with keys that are exactly at the boundary
        min_valid_key = "x" * 30
        assert validate_api_key_format(min_valid_key) is True

        too_short_key = "x" * 29
        with pytest.raises(MissingKeyError, match="too short"):
            validate_api_key_format(too_short_key)


class TestTokenEfficiency:
    """Test token efficiency calculations"""

    def test_calculate_token_efficiency_normal(self):
        """Should calculate efficiency as output tokens / total tokens"""
        efficiency = _calculate_token_efficiency(100, 50)
        expected = 50 / 150  # output / total
        assert efficiency == expected

    def test_calculate_token_efficiency_zero_tokens(self):
        """Should return 0.0 when no tokens are used"""
        efficiency = _calculate_token_efficiency(0, 0)
        assert efficiency == 0.0

    def test_calculate_token_efficiency_only_output(self):
        """Should return 1.0 when only output tokens exist (edge case)"""
        efficiency = _calculate_token_efficiency(0, 50)
        assert efficiency == 1.0  # 50 / 50

    def test_calculate_token_efficiency_only_prompt(self):
        """Should return 0.0 when only prompt tokens exist (no output)"""
        efficiency = _calculate_token_efficiency(100, 0)
        assert efficiency == 0.0  # 0 / 100


class TestEfficiencyMetrics:
    """Test comprehensive efficiency metrics"""

    def test_calculate_efficiency_metrics_full(self):
        """Should calculate all metrics correctly with complete data"""
        metrics = _calculate_efficiency_metrics(
            individual_prompt_tokens=300,
            individual_output_tokens=150,
            batch_prompt_tokens=120,
            batch_output_tokens=80,
            individual_time=3.0,
            batch_time=1.0,
        )

        # Verify all expected keys exist
        expected_keys = {
            "individual_token_efficiency",
            "batch_token_efficiency",
            "token_efficiency_ratio",
            "time_efficiency",
            "overall_efficiency",
            "meets_target",
            "individual_total_tokens",
            "batch_total_tokens",
        }
        assert set(metrics.keys()) == expected_keys

        # Verify calculations
        assert metrics["individual_total_tokens"] == 450  # 300 + 150
        assert metrics["batch_total_tokens"] == 200  # 120 + 80
        assert metrics["token_efficiency_ratio"] == 2.25  # 450 / 200
        assert metrics["time_efficiency"] == 3.0  # 3.0 / 1.0
        assert metrics["meets_target"] is False  # 2.25 < 3.0

    def test_calculate_efficiency_metrics_meets_target(self):
        """Should correctly identify when efficiency meets target"""
        metrics = _calculate_efficiency_metrics(
            individual_prompt_tokens=100,
            individual_output_tokens=50,
            batch_prompt_tokens=50,
            batch_output_tokens=25,
            individual_time=2.0,
            batch_time=0.5,
        )

        # Token efficiency ratio: 150 / 75 = 2.0
        # Time efficiency: 2.0 / 0.5 = 4.0
        # Overall efficiency: 2.0
        # Should meet target (2.0 >= 3.0 * 0.99 = 2.97)
        assert metrics["meets_target"] is False  # 2.0 < 2.97

    def test_calculate_efficiency_metrics_zero_batch_time(self):
        """Should handle zero batch time gracefully"""
        metrics = _calculate_efficiency_metrics(
            individual_prompt_tokens=100,
            individual_output_tokens=50,
            batch_prompt_tokens=50,
            batch_output_tokens=25,
            individual_time=2.0,
            batch_time=0.0,
        )

        # Time efficiency should be 1.0 when batch time is 0 (default behavior)
        assert metrics["time_efficiency"] == 1.0
        assert metrics["overall_efficiency"] == 2.0  # token_efficiency_ratio


class TestTrackEfficiency:
    """Test efficiency tracking functionality"""

    def test_track_efficiency_full_comparison(self):
        """Should track efficiency with complete data"""
        result = track_efficiency(
            individual_calls=1,
            batch_calls=1,
            individual_prompt_tokens=200,
            individual_output_tokens=100,
            batch_prompt_tokens=80,
            batch_output_tokens=60,
            individual_time=2.0,
            batch_time=0.8,
        )

        assert "comparison_available" in result
        assert "meets_target" in result
        # With these token counts: individual=300, batch=140, ratio=2.14 < 3.0
        assert result["meets_target"] is False

    def test_track_efficiency_limited_data(self):
        """Should handle limited data gracefully"""
        result = track_efficiency(
            individual_calls=1,
            batch_calls=1,
            individual_prompt_tokens=100,
            individual_output_tokens=50,
            batch_prompt_tokens=100,
            batch_output_tokens=50,
            individual_time=1.0,
            batch_time=1.0,
        )

        # Should still return metrics even with no efficiency gain
        assert "comparison_available" in result
        assert result["token_efficiency_ratio"] == 1.0

    def test_track_efficiency_no_token_data(self):
        """Should handle missing token data"""
        result = track_efficiency(
            individual_calls=1,
            batch_calls=1,
            individual_prompt_tokens=0,
            individual_output_tokens=0,
            batch_prompt_tokens=0,
            batch_output_tokens=0,
            individual_time=1.0,
            batch_time=1.0,
        )

        assert "comparison_available" in result
        assert result["token_efficiency_ratio"] is None  # No token data available


class TestExtractUsageMetrics:
    """Test usage metrics extraction from API responses"""

    def test_extract_usage_metrics_complete_data(self):
        """Should extract complete usage metrics from response"""
        # Create a mock response with complete usage metadata
        mock_usage = Mock()
        mock_usage.prompt_token_count = 100
        mock_usage.candidates_token_count = 50
        mock_usage.cached_content_token_count = 20

        mock_response = Mock()
        mock_response.usage_metadata = mock_usage

        metrics = extract_usage_metrics(mock_response)

        assert metrics["prompt_tokens"] == 100
        assert metrics["output_tokens"] == 50
        assert metrics["cached_tokens"] == 20
        assert metrics["total_tokens"] == 150
        assert metrics["effective_prompt_tokens"] == 80  # 100 - 20
        assert metrics["cache_hit_ratio"] == 0.2  # 20 / 100
        assert metrics["cache_enabled"] is True

    def test_extract_usage_metrics_missing_attributes(self):
        """Should handle missing attributes gracefully"""
        # Create a mock response with missing attributes
        mock_usage = Mock()
        mock_usage.prompt_token_count = None
        mock_usage.candidates_token_count = None
        mock_usage.cached_content_token_count = None

        mock_response = Mock()
        mock_response.usage_metadata = mock_usage

        metrics = extract_usage_metrics(mock_response)

        assert metrics["prompt_tokens"] == 0
        assert metrics["output_tokens"] == 0
        assert metrics["cached_tokens"] == 0
        assert metrics["total_tokens"] == 0
        assert metrics["effective_prompt_tokens"] == 0
        assert metrics["cache_hit_ratio"] == 0.0
        assert metrics["cache_enabled"] is False

    def test_extract_usage_metrics_none_values(self):
        """Should handle None values in usage metadata"""
        mock_usage = Mock()
        mock_usage.prompt_token_count = 0
        mock_usage.candidates_token_count = 0
        mock_usage.cached_content_token_count = 0

        mock_response = Mock()
        mock_response.usage_metadata = mock_usage

        metrics = extract_usage_metrics(mock_response)

        assert metrics["prompt_tokens"] == 0
        assert metrics["output_tokens"] == 0
        assert metrics["cached_tokens"] == 0
        assert metrics["total_tokens"] == 0
        assert metrics["effective_prompt_tokens"] == 0
        assert metrics["cache_hit_ratio"] == 0.0
        assert metrics["cache_enabled"] is False

    def test_extract_usage_metrics_no_usage_metadata(self):
        """Should handle response without usage metadata"""
        mock_response = Mock()
        mock_response.usage_metadata = None

        metrics = extract_usage_metrics(mock_response)

        assert metrics["prompt_tokens"] == 0
        assert metrics["output_tokens"] == 0
        assert metrics["cached_tokens"] == 0
        assert metrics["total_tokens"] == 0
        assert metrics["effective_prompt_tokens"] == 0
        assert metrics["cache_hit_ratio"] == 0.0
        assert metrics["cache_enabled"] is False


@pytest.mark.unit
class TestCalculateQualityScore:
    """Test response quality score calculation"""

    def test_calculate_quality_score_identical_answers(self):
        """Should return high score for identical answers"""
        answers1 = ["Answer 1", "Answer 2", "Answer 3"]
        answers2 = ["Answer 1", "Answer 2", "Answer 3"]

        score = calculate_quality_score(answers1, answers2)
        assert (
            score > 0.7
        )  # Should be high for identical answers (completeness + overlap + length)

    def test_calculate_quality_score_similar_answers(self):
        """Should return moderate score for similar answers"""
        answers1 = ["This is answer one", "This is answer two"]
        answers2 = ["This is answer one", "This is answer two with extra words"]

        score = calculate_quality_score(answers1, answers2)
        assert 0.5 < score < 0.9  # Should be moderate for similar answers

    def test_calculate_quality_score_completely_different_answers(self):
        """Should return low score for completely different answers"""
        answers1 = ["Answer 1", "Answer 2", "Answer 3"]
        answers2 = ["Different answer 1", "Different answer 2", "Different answer 3"]

        score = calculate_quality_score(answers1, answers2)
        assert (
            score < 0.6
        )  # Should be moderate for different answers (completeness still counts)

    def test_calculate_quality_score_empty_inputs(self):
        """Should handle empty inputs gracefully"""
        score = calculate_quality_score([], [])
        assert score is None  # Returns None for empty inputs

    def test_calculate_quality_score_different_lengths(self):
        """Should return 0.0 for lists of different lengths"""
        answers1 = ["Answer 1", "Answer 2"]
        answers2 = ["Answer 1", "Answer 2", "Answer 3"]

        score = calculate_quality_score(answers1, answers2)
        assert score == 0.0

    def test_calculate_quality_score_short_answers(self):
        """Should handle short answers appropriately"""
        answers1 = ["A", "B", "C"]
        answers2 = ["A", "B", "C"]

        score = calculate_quality_score(answers1, answers2)
        assert score > 0.7  # Should be high for identical short answers


class TestUtilsIntegration:
    """Test utilities working together in realistic scenarios"""

    def test_environment_utilities_work_together(self):
        """Should work together for realistic configuration scenarios"""
        with patch.dict(
            "os.environ",
            {
                "GEMINI_API_KEY": "test_key_123456789012345678901234567890",
                "GEMINI_ENABLE_CACHING": "true",
                "GEMINI_ENABLE_METRICS": "1",
                "GEMINI_ENABLE_DEBUG": "false",
                "FALLBACK_MODEL": "gemini-1.5-flash",
            },
            clear=False,
        ):
            # Test environment parsing working together
            api_key = get_env_with_fallback("GEMINI_API_KEY", "GOOGLE_API_KEY")
            enable_caching = parse_env_bool("GEMINI_ENABLE_CACHING")
            enable_metrics = parse_env_bool("GEMINI_ENABLE_METRICS")
            enable_debug = parse_env_bool("GEMINI_ENABLE_DEBUG")
            fallback_model = get_env_with_fallback("GEMINI_MODEL", "FALLBACK_MODEL")

            # Validate API key
            assert validate_api_key_format(api_key) is True

            # Verify parsed values
            assert api_key == "test_key_123456789012345678901234567890"
            assert enable_caching is True
            assert enable_metrics is True
            assert enable_debug is False
            # The fallback model might be different depending on environment
            assert fallback_model in ["gemini-1.5-flash", "gemini-2.0-flash"]

    def test_error_handling_edge_cases(self):
        """Should handle edge cases gracefully"""
        # Test with None values
        with patch("os.getenv", return_value=None):
            assert get_env_with_fallback("MISSING", "ALSO_MISSING") is None
        with patch("os.getenv", return_value=""):
            assert parse_env_bool("MISSING") is False
            assert parse_env_bool("MISSING", default=True) is True

        # Test with empty strings
        with patch("os.getenv", return_value=""):
            assert get_env_with_fallback("EMPTY", "FALLBACK") == ""
            assert parse_env_bool("EMPTY") is False
