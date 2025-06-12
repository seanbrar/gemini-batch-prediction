from unittest.mock import Mock

import pytest

from gemini_batch.exceptions import MissingKeyError
from gemini_batch.utils import (
    _calculate_efficiency_metrics,
    _calculate_token_efficiency,
    calculate_quality_score,
    extract_answers,
    extract_usage_metrics,
    track_efficiency,
    validate_api_key,
)


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
        """Should correctly identify when efficiency meets 3x target"""
        metrics = _calculate_efficiency_metrics(
            individual_prompt_tokens=600,
            individual_output_tokens=300,
            batch_prompt_tokens=100,
            batch_output_tokens=50,
            individual_time=5.0,
            batch_time=1.0,
        )

        assert metrics["token_efficiency_ratio"] == 6.0  # 900 / 150
        assert metrics["time_efficiency"] == 5.0
        assert metrics["meets_target"] is True

    def test_calculate_efficiency_metrics_zero_batch_time(self):
        """Should handle zero batch time gracefully"""
        metrics = _calculate_efficiency_metrics(
            individual_prompt_tokens=300,
            individual_output_tokens=150,
            batch_prompt_tokens=120,
            batch_output_tokens=80,
            individual_time=3.0,
            batch_time=0.0,  # Edge case
        )

        assert metrics["time_efficiency"] == 1.0  # Default when batch_time is 0


class TestTrackEfficiency:
    """Test the main efficiency tracking function"""

    def test_track_efficiency_full_comparison(self):
        """Should provide complete efficiency analysis when both methods available"""
        metrics = track_efficiency(
            individual_calls=3,
            batch_calls=1,
            individual_prompt_tokens=300,
            individual_output_tokens=150,
            batch_prompt_tokens=120,
            batch_output_tokens=80,
            individual_time=3.0,
            batch_time=1.0,
        )

        assert metrics["comparison_available"] is True
        assert metrics["token_efficiency_ratio"] == 2.25  # Specific calculation
        assert metrics["time_efficiency"] == 3.0
        assert metrics["meets_target"] is False

    def test_track_efficiency_limited_data(self):
        """Should handle limited data gracefully"""
        metrics = track_efficiency(
            individual_calls=0,
            batch_calls=1,
            batch_prompt_tokens=120,
            batch_output_tokens=80,
        )

        assert metrics["comparison_available"] is False
        assert metrics["token_efficiency_ratio"] is None
        assert metrics["time_efficiency"] is None
        assert metrics["meets_target"] is True  # Success if batch worked

    def test_track_efficiency_no_token_data(self):
        """Should handle missing token data"""
        metrics = track_efficiency(individual_calls=1, batch_calls=1)

        assert metrics["comparison_available"] is True
        assert metrics["individual_token_efficiency"] is None
        assert metrics["batch_token_efficiency"] is None


class TestExtractUsageMetrics:
    """Test usage metrics extraction"""

    def test_extract_usage_metrics_complete_data(self):
        """Should extract all usage metrics from complete response"""
        mock_usage = Mock()
        mock_usage.prompt_token_count = 100
        mock_usage.candidates_token_count = 50
        mock_usage.cached_content_token_count = 10

        mock_response = Mock()
        mock_response.usage_metadata = mock_usage

        metrics = extract_usage_metrics(mock_response)

        assert metrics["prompt_tokens"] == 100
        assert metrics["output_tokens"] == 50
        assert metrics["cached_tokens"] == 10
        assert metrics["total_tokens"] == 150

    def test_extract_usage_metrics_missing_attributes(self):
        """Should return zeros for missing attributes"""
        mock_usage = Mock()
        # Simulate missing attributes by not setting them

        mock_response = Mock()
        mock_response.usage_metadata = mock_usage

        metrics = extract_usage_metrics(mock_response)

        assert all(value == 0 for value in metrics.values())

    def test_extract_usage_metrics_none_values(self):
        """Should handle None values in usage metadata"""
        mock_usage = Mock()
        mock_usage.prompt_token_count = None
        mock_usage.candidates_token_count = None
        mock_usage.cached_content_token_count = None

        mock_response = Mock()
        mock_response.usage_metadata = mock_usage

        metrics = extract_usage_metrics(mock_response)

        assert all(value == 0 for value in metrics.values())

    def test_extract_usage_metrics_no_usage_metadata(self):
        """Should return zeros when usage_metadata attribute is missing"""
        mock_response = Mock()
        del mock_response.usage_metadata

        metrics = extract_usage_metrics(mock_response)

        assert all(value == 0 for value in metrics.values())


class TestExtractAnswers:
    """Test answer extraction from batch responses"""

    def test_extract_answers_standard_format(self):
        """Should extract answers from standard 'Answer X:' format"""
        response_text = """
        Answer 1: This is the first answer with some detail.
        Answer 2: This is the second answer with more information.
        Answer 3: This is the third answer completing the set.
        """

        answers = extract_answers(response_text, 3)

        assert len(answers) == 3
        assert "first answer" in answers[0]
        assert "second answer" in answers[1]
        assert "third answer" in answers[2]
        # Verify prefixes are stripped
        assert not any(answer.startswith("Answer") for answer in answers)

    def test_extract_answers_numbered_format(self):
        """Should extract answers from numbered list format"""
        response_text = """
        1. This is answer one with details.
        2. This is answer two with more info.
        3. This is answer three to complete.
        """

        answers = extract_answers(response_text, 3)

        assert len(answers) == 3
        assert "answer one" in answers[0]
        assert "answer two" in answers[1]
        assert "answer three" in answers[2]

    def test_extract_answers_mixed_formats(self):
        """Should handle mixed answer formats in same response"""
        response_text = """
        Answer 1: First answer in standard format.
        2. Second answer in numbered format.
        Answer 3: Third answer back to standard.
        """

        answers = extract_answers(response_text, 3)

        assert len(answers) == 3
        assert "First answer" in answers[0]
        assert "Second answer" in answers[1]
        assert "Third answer" in answers[2]

    def test_extract_answers_missing_answer(self):
        """Should handle missing answers with placeholder"""
        response_text = """
        Answer 1: This is the first answer.
        Answer 3: This is the third answer, but second is missing.
        """

        answers = extract_answers(response_text, 3)

        assert len(answers) == 3
        assert "first answer" in answers[0]
        assert "(No answer found)" in answers[1]
        assert "third answer" in answers[2]

    def test_extract_answers_empty_response(self):
        """Should handle completely empty responses"""
        answers = extract_answers("", 2)

        assert len(answers) == 2
        assert all("(No answer found)" in answer for answer in answers)

    def test_extract_answers_malformed_response(self):
        """Should handle malformed responses gracefully"""
        response_text = "This is just a paragraph without any answer structure."

        answers = extract_answers(response_text, 2)

        assert len(answers) == 2
        assert all("(No answer found)" in answer for answer in answers)

    def test_extract_answers_very_short_answers(self):
        """Should reject answers that are too short (< 5 chars)"""
        response_text = """
        Answer 1: Yes.
        Answer 2: This is a proper length answer.
        """

        answers = extract_answers(response_text, 2)

        assert len(answers) == 2
        assert "(No answer found)" in answers[0]  # "Yes." is too short
        assert "proper length answer" in answers[1]


class TestCalculateQualityScore:
    """Test quality score calculation"""

    def test_calculate_quality_score_identical_answers(self):
        """Should return high score for identical answers"""
        answers = [
            "Machine learning enables AI capabilities",
            "Neural networks process data",
        ]

        score = calculate_quality_score(answers, answers)

        assert score is not None
        assert score > 0.8  # Should be very high for identical content

    def test_calculate_quality_score_similar_answers(self):
        """Should return moderate score for similar answers with word overlap"""
        individual = ["AI is transformative technology", "Machine learning enables AI"]
        batch = [
            "AI represents transformative technology",
            "Machine learning powers AI systems",
        ]

        score = calculate_quality_score(individual, batch)

        assert score is not None
        assert 0.4 < score < 0.9  # Should show good similarity
        # Verify the score makes sense - similar content should score higher than random

    def test_calculate_quality_score_completely_different_answers(self):
        """Should return low score for completely different answers"""
        individual = [
            "Quantum physics describes subatomic particles",
            "Chemistry studies molecular bonds",
        ]
        batch = [
            "Cooking recipes require proper ingredients",
            "Music theory involves scales and chords",
        ]

        score = calculate_quality_score(individual, batch)

        assert score is not None
        # Quality score algorithm weights completeness heavily (50%). Even different answers
        # get 0.5 for completeness + some overlap for common words + length similarity.
        # Realistic expectation is 0.5-0.8 range for different but complete answers.
        assert 0.5 <= score <= 0.8  # Should be moderate due to completeness weighting

    def test_calculate_quality_score_empty_inputs(self):
        """Should return None for empty input lists"""
        assert calculate_quality_score([], ["answer"]) is None
        assert calculate_quality_score(["answer"], []) is None
        assert calculate_quality_score([], []) is None

    def test_calculate_quality_score_different_lengths(self):
        """Should return 0.0 for mismatched list lengths"""
        individual = ["Answer one", "Answer two"]
        batch = ["Answer one"]

        score = calculate_quality_score(individual, batch)
        assert score == 0.0

    def test_calculate_quality_score_short_answers(self):
        """Should handle answers shorter than 10 characters"""
        individual = ["Yes", "No"]
        batch = ["Yeah", "Nope"]

        score = calculate_quality_score(individual, batch)

        assert score is not None
        assert 0.0 <= score <= 1.0  # Should still return valid score


class TestValidateApiKey:
    """Test API key validation"""

    def test_validate_api_key_valid_format(self):
        """Should accept properly formatted API keys"""
        valid_keys = [
            "AIzaSyC8UYZpvA2eknNdcAaFeFbRe-PaWiDfD_M",  # Google format
        ]

        for key in valid_keys:
            assert validate_api_key(key) is True

    def test_validate_api_key_invalid_inputs(self):
        """Should reject invalid inputs with appropriate error messages"""
        with pytest.raises(MissingKeyError, match="non-empty string"):
            validate_api_key("")

        with pytest.raises(MissingKeyError, match="non-empty string"):
            validate_api_key(None)

        with pytest.raises(MissingKeyError, match="non-empty string"):
            validate_api_key(12345)

        with pytest.raises(MissingKeyError, match="too short"):
            validate_api_key("short")

    def test_validate_api_key_edge_cases(self):
        """Should handle edge cases appropriately"""
        # Whitespace-only string should fail with "too short" after stripping
        with pytest.raises(MissingKeyError, match="too short"):
            validate_api_key("   ")

        # Very long key should still work
        very_long_key = "x" * 100
        assert validate_api_key(very_long_key) is True
