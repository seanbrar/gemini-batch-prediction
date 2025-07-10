"""
API tests for structured output functionality with real API calls.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from tests.fixtures.content_samples import EDUCATIONAL_CONTENT


class TestAPIStructuredOutput:
    """Test structured output functionality with real API calls."""

    class ResearchSummary(BaseModel):
        """Simple research summary model for testing."""

        main_topic: str
        key_findings: List[str]
        methodology: str
        confidence: float = Field(ge=0.0, le=1.0)

    class DocumentAnalysis(BaseModel):
        """Document analysis model for testing."""

        title: str
        summary: str
        key_points: List[str]
        sentiment: str
        complexity_level: str

    class ComparisonResult(BaseModel):
        """Comparison result model for testing."""

        similarities: List[str]
        differences: List[str]
        overall_assessment: str
        recommendation: Optional[str] = None

    def test_pydantic_schema_is_populated_correctly(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test that Pydantic schema is populated correctly with real API."""
        content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
        questions = ["Summarize the research in structured format"]

        result = real_batch_processor.process_questions(
            content, questions, response_schema=self.ResearchSummary
        )

        # Verify structured data is present
        assert "structured_data" in result
        assert len(result["structured_data"]) == 1

        # Verify structured data is valid
        structured_data = result["structured_data"][0]
        assert isinstance(structured_data, self.ResearchSummary)
        assert len(structured_data.main_topic) > 0
        assert len(structured_data.key_findings) > 0
        assert len(structured_data.methodology) > 0
        assert 0.0 <= structured_data.confidence <= 1.0

        print("✅ Pydantic schema population test passed")
        print(f"   Main topic: {structured_data.main_topic}")
        print(f"   Key findings: {len(structured_data.key_findings)} items")
        print(f"   Confidence: {structured_data.confidence}")

    def test_multiple_questions_and_sources_structured_output(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test structured output with multiple questions and sources."""
        sources = [
            EDUCATIONAL_CONTENT["short_lesson"],
            EDUCATIONAL_CONTENT["medium_article"],
        ]
        questions = ["Analyze the first source", "Compare the two sources"]

        result = real_batch_processor.process_questions_multi_source(
            sources, questions, response_schema=self.DocumentAnalysis
        )

        # Verify structured data for multiple questions
        assert "structured_data" in result
        assert len(result["structured_data"]) == 2

        # Verify each structured data is valid
        for i, structured_data in enumerate(result["structured_data"]):
            assert isinstance(structured_data, self.DocumentAnalysis)
            assert len(structured_data.title) > 0
            assert len(structured_data.summary) > 0
            assert len(structured_data.key_points) > 0
            assert structured_data.sentiment in ["positive", "negative", "neutral"]
            assert structured_data.complexity_level in [
                "basic",
                "intermediate",
                "advanced",
            ]

        print("✅ Multiple questions structured output test passed")
        print(f"   Questions processed: {len(result['structured_data'])}")

    def test_validation_errors_in_structured_output(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test handling of validation errors in structured output."""

        # Use a model with strict constraints
        class StrictModel(BaseModel):
            required_field: str
            numeric_field: int = Field(ge=0, le=100)
            enum_field: str = Field(pattern="^(option1|option2|option3)$")

        content = "This is test content for validation testing."
        questions = ["Analyze this content"]

        # This should handle validation errors gracefully
        result = real_batch_processor.process_questions(
            content, questions, response_schema=StrictModel
        )

        # Should still return results, even if validation fails
        assert "answers" in result
        assert len(result["answers"]) == 1
        assert len(result["answers"][0]) > 0

        # May or may not have structured data depending on validation
        if "structured_data" in result:
            print("✅ Validation error handling test passed")
            print("   Structured data was generated despite constraints")
        else:
            print("✅ Validation error handling test passed")
            print("   Graceful fallback to text answers")

    def test_optional_fields_in_structured_output(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test structured output with optional fields."""
        content = EDUCATIONAL_CONTENT["medium_article"]
        questions = ["Analyze this content and provide recommendations"]

        result = real_batch_processor.process_questions(
            content, questions, response_schema=self.ComparisonResult
        )

        # Verify structured data is present
        assert "structured_data" in result
        assert len(result["structured_data"]) == 1

        structured_data = result["structured_data"][0]
        assert isinstance(structured_data, self.ComparisonResult)
        assert len(structured_data.similarities) > 0
        assert len(structured_data.differences) > 0
        assert len(structured_data.overall_assessment) > 0
        # recommendation is optional, so it may or may not be present
        assert hasattr(structured_data, "recommendation")

        print("✅ Optional fields structured output test passed")
        print(f"   Similarities: {len(structured_data.similarities)} items")
        print(f"   Differences: {len(structured_data.differences)} items")

    def test_performance_of_structured_output(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test performance of structured output vs regular output."""
        import time

        content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
        questions = ["Summarize the research"]

        # Test with structured output
        start_time = time.time()
        structured_result = real_batch_processor.process_questions(
            content, questions, response_schema=self.ResearchSummary
        )
        structured_time = time.time() - start_time

        # Test without structured output
        start_time = time.time()
        regular_result = real_batch_processor.process_questions(content, questions)
        regular_time = time.time() - start_time

        # Verify both succeeded
        assert "answers" in structured_result
        assert "answers" in regular_result
        assert len(structured_result["answers"]) == 1
        assert len(regular_result["answers"]) == 1

        # Verify structured data was generated
        assert "structured_data" in structured_result
        assert len(structured_result["structured_data"]) == 1

        print("✅ Structured output performance test passed")
        print(f"   Structured time: {structured_time:.2f}s")
        print(f"   Regular time: {regular_time:.2f}s")
        print(f"   Performance ratio: {structured_time / regular_time:.2f}x")

    def test_complex_schemas_in_structured_output(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test complex schemas with nested models and constraints."""

        class Author(BaseModel):
            name: str
            affiliation: str
            expertise: List[str]

        class Publication(BaseModel):
            title: str
            authors: List[Author]
            year: int = Field(ge=1900, le=2024)
            citations: int = Field(ge=0)
            impact_factor: Optional[float] = None

        content = EDUCATIONAL_CONTENT["academic_paper_excerpt"]
        questions = ["Extract publication information"]

        result = real_batch_processor.process_questions(
            content, questions, response_schema=Publication
        )

        # Verify structured data is present
        assert "structured_data" in result
        assert len(result["structured_data"]) == 1

        structured_data = result["structured_data"][0]
        assert isinstance(structured_data, Publication)
        assert len(structured_data.title) > 0
        assert len(structured_data.authors) > 0

        # Verify nested author data
        for author in structured_data.authors:
            assert len(author.name) > 0
            assert len(author.affiliation) > 0
            assert len(author.expertise) > 0

        # Verify constraints
        assert 1900 <= structured_data.year <= 2024
        assert structured_data.citations >= 0

        print("✅ Complex schema structured output test passed")
        print(f"   Title: {structured_data.title}")
        print(f"   Authors: {len(structured_data.authors)}")
        print(f"   Year: {structured_data.year}")

    def test_error_recovery_in_structured_output(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test error recovery in structured output processing."""

        # Use a model that might be challenging for the API
        class ChallengingModel(BaseModel):
            mathematical_formula: str
            technical_specifications: List[str]
            performance_metrics: dict

        content = (
            "This is a simple test content that may not contain complex technical data."
        )
        questions = ["Extract technical specifications"]

        # Should handle gracefully even if the model is challenging
        result = real_batch_processor.process_questions(
            content, questions, response_schema=ChallengingModel
        )

        # Should still return results
        assert "answers" in result
        assert len(result["answers"]) == 1
        assert len(result["answers"][0]) > 0

        # May or may not have structured data
        if "structured_data" in result:
            print("✅ Error recovery test passed - structured data generated")
        else:
            print("✅ Error recovery test passed - graceful fallback")

    def test_structured_output_with_different_content_types(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test structured output with different content types."""

        class ContentAnalysis(BaseModel):
            content_type: str
            main_themes: List[str]
            readability_score: float = Field(ge=0.0, le=1.0)
            technical_complexity: str

        # Test with different content types
        content_types = [
            EDUCATIONAL_CONTENT["short_lesson"],
            EDUCATIONAL_CONTENT["medium_article"],
            EDUCATIONAL_CONTENT["academic_paper_excerpt"],
        ]

        results = []
        for content in content_types:
            result = real_batch_processor.process_questions(
                content, ["Analyze this content"], response_schema=ContentAnalysis
            )
            results.append(result)

        # Verify all results have structured data
        for i, result in enumerate(results):
            assert "structured_data" in result
            assert len(result["structured_data"]) == 1

            structured_data = result["structured_data"][0]
            assert isinstance(structured_data, ContentAnalysis)
            assert len(structured_data.content_type) > 0
            assert len(structured_data.main_themes) > 0
            assert 0.0 <= structured_data.readability_score <= 1.0
            assert len(structured_data.technical_complexity) > 0

        print("✅ Different content types structured output test passed")
        print(f"   Content types processed: {len(results)}")

    def test_structured_output_consistency(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test consistency of structured output across multiple calls."""
        content = EDUCATIONAL_CONTENT["short_lesson"]
        questions = ["Summarize the key points"]

        # Make multiple calls with same content and schema
        results = []
        for i in range(3):
            result = real_batch_processor.process_questions(
                content, questions, response_schema=self.ResearchSummary
            )
            results.append(result)

        # Verify all results have structured data
        for i, result in enumerate(results):
            assert "structured_data" in result
            assert len(result["structured_data"]) == 1

            structured_data = result["structured_data"][0]
            assert isinstance(structured_data, self.ResearchSummary)
            assert len(structured_data.main_topic) > 0
            assert len(structured_data.key_findings) > 0

        print("✅ Structured output consistency test passed")
        print(f"   Calls made: {len(results)}")
        print(
            f"   All calls successful: {all('structured_data' in r for r in results)}"
        )
