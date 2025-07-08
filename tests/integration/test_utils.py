import json
from unittest.mock import patch

from gemini_batch.efficiency.tracking import track_efficiency
from gemini_batch.response.processor import ResponseProcessor
from gemini_batch.utils import (
    get_env_with_fallback,
    parse_env_bool,
    validate_api_key_format,
)


class TestUtilityIntegration:
    """Test utilities working together in realistic scenarios"""

    def test_environment_utilities_work_together_in_configuration(self):
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
            # Remove GEMINI_MODEL if it exists to test fallback
            import os

            if "GEMINI_MODEL" in os.environ:
                del os.environ["GEMINI_MODEL"]

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
            assert fallback_model == "gemini-1.5-flash"

    def test_efficiency_tracking_with_realistic_scenarios(self):
        """Should track efficiency with realistic token usage patterns"""
        # Simulate realistic batch vs individual comparison
        individual_calls = 2
        batch_calls = 1
        individual_prompt_tokens = 800  # Higher due to repetition
        individual_output_tokens = 200
        batch_prompt_tokens = 500
        batch_output_tokens = 200
        batch_time = 2.5
        individual_time = 5.0

        # Track efficiency
        efficiency_data = track_efficiency(
            individual_calls=individual_calls,
            batch_calls=batch_calls,
            individual_prompt_tokens=individual_prompt_tokens,
            individual_output_tokens=individual_output_tokens,
            batch_prompt_tokens=batch_prompt_tokens,
            batch_output_tokens=batch_output_tokens,
            batch_time=batch_time,
            individual_time=individual_time,
        )

        # Should show batch is more efficient
        assert efficiency_data["time_efficiency"] > 1.0  # Batch faster
        assert (
            efficiency_data["token_efficiency_ratio"] > 1.0
        )  # Batch uses fewer tokens
        assert efficiency_data["comparison_available"] is True

    def test_answer_extraction_handles_various_formats_robustly(self):
        """Should robustly handle various answer formats that might occur in practice"""
        # Test realistic mixed format responses
        mixed_response = {
            "text": json.dumps(
                [
                    "This is a detailed response about the first topic with multiple sentences and technical details.",
                    "For the second question, I'll provide this answer in a different format but still meaningful.",
                    "The third response is also comprehensive and provides useful information.",
                ]
            )
        }

        processor = ResponseProcessor()
        result = processor.process_response(mixed_response, expected_questions=3)

        assert len(result.answers) == 3
        assert "detailed response about the first topic" in result.answers[0]
        assert "second question" in result.answers[1]
        assert "third response is also comprehensive" in result.answers[2]

        # Verify all answers are substantive (not just "No answer found")
        assert all("No answer found" not in answer for answer in result.answers)

    def test_complete_workflow_integration(self):
        """Should handle complete workflow from environment setup through efficiency tracking"""
        # Test a complete integration scenario
        with patch.dict(
            "os.environ",
            {
                "GEMINI_API_KEY": "workflow_key_123456789012345678901234567890",
                "GEMINI_ENABLE_METRICS": "true",
            },
            clear=False,
        ):
            # 1. Environment configuration
            api_key = get_env_with_fallback("GEMINI_API_KEY", "GOOGLE_API_KEY")
            enable_metrics = parse_env_bool("GEMINI_ENABLE_METRICS")

            # 2. API key validation
            assert validate_api_key_format(api_key) is True

            # 3. Simulate processing and answer extraction
            batch_response = json.dumps(
                [
                    "Machine learning enables pattern recognition in data.",
                    "AI systems can process natural language effectively.",
                    "Neural networks form the backbone of modern AI.",
                ]
            )

            processor = ResponseProcessor()
            result = processor.process_response(
                {"text": batch_response}, expected_questions=3
            )
            assert len(result.answers) == 3

            # 4. Track efficiency (if metrics enabled)
            if enable_metrics:
                efficiency_data = track_efficiency(
                    individual_calls=3,
                    batch_calls=1,
                    individual_prompt_tokens=600,
                    individual_output_tokens=150,
                    batch_prompt_tokens=400,
                    batch_output_tokens=150,
                    batch_time=2.0,
                    individual_time=4.5,
                )

                # Verify the complete workflow produced valid efficiency data
                assert efficiency_data["comparison_available"] is True
                assert efficiency_data["time_efficiency"] > 1.0
