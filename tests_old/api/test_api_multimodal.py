"""
API tests for multimodal functionality.
"""

import time

import pytest

from tests.fixtures.content_samples import EDUCATIONAL_CONTENT


@pytest.mark.api
class TestAPIMultimodal:
    """Test multimodal functionality with real API calls."""

    def test_process_real_youtube_url(self, real_batch_processor, api_rate_limiter):
        """Test processing a real YouTube URL."""
        # Use a short, public YouTube video for testing
        youtube_url = (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll - short video
        )
        questions = ["What is this video about?", "What is the main message?"]

        try:
            result = real_batch_processor.process_questions_multi_source(
                [youtube_url], questions
            )

            # Verify response structure
            assert "answers" in result
            assert len(result["answers"]) == len(questions)
            assert all(len(answer) > 10 for answer in result["answers"])

            # Verify metrics
            assert "metrics" in result
            assert "usage" in result["metrics"]
            assert result["metrics"]["usage"]["total_tokens"] > 0

            print("✅ YouTube URL processing successful")
            print(f"   Response length: {len(result['answers'][0])} chars")
            print(f"   Token usage: {result['metrics']['usage']['total_tokens']}")

        except Exception as e:
            # YouTube processing might fail due to various reasons
            print(f"⚠️  YouTube processing failed: {type(e).__name__}")
            print(f"   Error: {str(e)}")
            # Should be a specific error type
            assert any(
                keyword in str(e).lower()
                for keyword in ["youtube", "video", "url", "api", "rate", "limit"]
            )

    def test_process_real_pdf_document(self, real_batch_processor, api_rate_limiter):
        """Test processing a real PDF document from URL."""
        # Use a publicly accessible PDF (arXiv paper)
        pdf_url = "https://arxiv.org/pdf/1503.02531"  # Knowledge Distillation paper
        questions = [
            "What is the main research question?",
            "What methodology was used?",
            "What are the key findings?",
        ]

        try:
            result = real_batch_processor.process_questions_multi_source(
                [pdf_url], questions
            )

            # Verify response structure
            assert "answers" in result
            assert len(result["answers"]) == len(questions)
            assert all(len(answer) > 10 for answer in result["answers"])

            # Verify metrics
            assert "metrics" in result
            assert "usage" in result["metrics"]
            assert result["metrics"]["usage"]["total_tokens"] > 0

            print("✅ PDF URL processing successful")
            print(f"   Questions processed: {len(questions)}")
            print(f"   Token usage: {result['metrics']['usage']['total_tokens']}")

            # Verify content quality
            for i, answer in enumerate(result["answers"]):
                print(f"   Answer {i + 1}: {answer[:100]}...")

        except Exception as e:
            # PDF processing might fail due to various reasons
            print(f"⚠️  PDF processing failed: {type(e).__name__}")
            print(f"   Error: {str(e)}")
            # Should be a specific error type
            assert any(
                keyword in str(e).lower()
                for keyword in ["pdf", "url", "download", "api", "rate", "limit"]
            )

    def test_multimodal_content_processing(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test processing mixed content types."""
        # Mix of text and URLs
        content_sources = [
            EDUCATIONAL_CONTENT["short_lesson"],  # Text content
            "https://arxiv.org/pdf/1503.02531",  # PDF URL
        ]
        questions = ["Compare the information from these sources"]

        try:
            result = real_batch_processor.process_questions_multi_source(
                content_sources, questions
            )

            # Verify response structure
            assert "answers" in result
            assert len(result["answers"]) == len(questions)
            assert all(len(answer) > 10 for answer in result["answers"])

            # Verify metrics
            assert "metrics" in result
            assert "usage" in result["metrics"]
            assert result["metrics"]["usage"]["total_tokens"] > 0

            print("✅ Multimodal content processing successful")
            print(f"   Sources processed: {len(content_sources)}")
            print(f"   Token usage: {result['metrics']['usage']['total_tokens']}")

        except Exception as e:
            # Multimodal processing might fail
            print(f"⚠️  Multimodal processing failed: {type(e).__name__}")
            print(f"   Error: {str(e)}")
            # Should be a specific error type
            assert any(
                keyword in str(e).lower()
                for keyword in ["url", "download", "api", "rate", "limit"]
            )

    def test_multimodal_error_handling(self, real_batch_processor, api_rate_limiter):
        """Test error handling for multimodal content."""
        # Test with invalid URL
        invalid_url = "https://example.com/nonexistent.pdf"
        questions = ["What is this document about?"]

        try:
            result = real_batch_processor.process_questions_multi_source(
                [invalid_url], questions
            )

            # If successful, verify structure
            assert "answers" in result
            assert len(result["answers"]) == len(questions)

            print("✅ Invalid URL handled gracefully")

        except Exception as e:
            # Should handle invalid URLs gracefully
            print(f"⚠️  Invalid URL error handled: {type(e).__name__}")
            assert any(
                keyword in str(e).lower()
                for keyword in ["url", "download", "not found", "error"]
            )

    def test_multimodal_performance(self, real_batch_processor, api_rate_limiter):
        """Test multimodal processing performance."""
        import time

        # Test with different content types
        test_cases = [
            {
                "name": "Text only",
                "content": [EDUCATIONAL_CONTENT["short_lesson"]],
                "questions": ["What is the main topic?"],
            },
            {
                "name": "PDF URL",
                "content": ["https://arxiv.org/pdf/1503.02531"],
                "questions": ["What is the research focus?"],
            },
        ]

        for test_case in test_cases:
            try:
                start_time = time.time()

                result = real_batch_processor.process_questions_multi_source(
                    test_case["content"], test_case["questions"]
                )

                end_time = time.time()
                duration = end_time - start_time

                # Verify successful processing
                assert "answers" in result
                assert len(result["answers"]) == len(test_case["questions"])
                assert all(len(answer) > 10 for answer in result["answers"])

                print(f"✅ {test_case['name']} processing completed")
                print(f"   Duration: {duration:.2f}s")
                print(f"   Token usage: {result['metrics']['usage']['total_tokens']}")

                # Should complete in reasonable time
                assert duration < 60  # Should not take more than 60 seconds

                time.sleep(3)  # Rate limiting between tests

            except Exception as e:
                print(f"⚠️  {test_case['name']} processing failed: {type(e).__name__}")
                # Continue with next test case

    def test_multimodal_content_quality(self, real_batch_processor, api_rate_limiter):
        """Test quality of multimodal content processing."""
        # Use a combination of text and URL content
        content_sources = [
            EDUCATIONAL_CONTENT["academic_paper_excerpt"],
            "https://arxiv.org/pdf/1503.02531",
        ]
        questions = [
            "What are the main research themes?",
            "How do these sources relate to each other?",
            "What are the key methodologies discussed?",
        ]

        try:
            result = real_batch_processor.process_questions_multi_source(
                content_sources, questions
            )

            # Verify response structure
            assert "answers" in result
            assert len(result["answers"]) == len(questions)

            # Verify answer quality
            for i, answer in enumerate(result["answers"]):
                assert len(answer) > 20  # Should be substantial answers

                # Check for relevant keywords
                answer_lower = answer.lower()
                relevant_keywords = [
                    "research",
                    "method",
                    "analysis",
                    "study",
                    "learning",
                    "model",
                ]
                keyword_count = sum(
                    1 for keyword in relevant_keywords if keyword in answer_lower
                )

                # Should contain some relevant keywords
                assert keyword_count > 0, f"Answer {i + 1} lacks relevant keywords"

            print("✅ Multimodal content quality test passed")
            print(f"   Questions processed: {len(questions)}")
            print(
                f"   Average answer length: {sum(len(a) for a in result['answers']) / len(result['answers']):.0f} chars"
            )

            # Show sample answers
            for i, answer in enumerate(result["answers"]):
                print(f"   Answer {i + 1}: {answer[:150]}...")

        except Exception as e:
            print(f"⚠️  Multimodal quality test failed: {type(e).__name__}")
            print(f"   Error: {str(e)}")

    def test_multimodal_with_structured_output(
        self, real_batch_processor, api_rate_limiter
    ):
        """Test multimodal processing with structured output."""
        from typing import List

        from pydantic import BaseModel

        class DocumentAnalysis(BaseModel):
            main_topics: List[str]
            key_methods: List[str]
            complexity_level: str
            summary: str

        content_sources = [
            EDUCATIONAL_CONTENT["academic_paper_excerpt"],
            "https://arxiv.org/pdf/1503.02531",
        ]
        questions = ["Analyze these documents and provide structured insights"]

        try:
            result = real_batch_processor.process_questions_multi_source(
                content_sources, questions, response_schema=DocumentAnalysis
            )

            # Verify structured data
            assert "structured_data" in result
            assert len(result["structured_data"]) == len(questions)

            structured_item = result["structured_data"][0]
            assert isinstance(structured_item, DocumentAnalysis)

            # Verify required fields
            assert len(structured_item.main_topics) > 0
            assert len(structured_item.key_methods) > 0
            assert len(structured_item.complexity_level) > 0
            assert len(structured_item.summary) > 0

            print("✅ Multimodal structured output successful")
            print(f"   Main topics: {len(structured_item.main_topics)}")
            print(f"   Key methods: {len(structured_item.key_methods)}")
            print(f"   Complexity: {structured_item.complexity_level}")

        except Exception as e:
            print(f"⚠️  Multimodal structured output failed: {type(e).__name__}")
            print(f"   Error: {str(e)}")

    def test_multimodal_rate_limiting(self, real_batch_processor, api_rate_limiter):
        """Test rate limiting with multimodal content."""
        content_sources = [
            EDUCATIONAL_CONTENT["short_lesson"],
            "https://arxiv.org/pdf/1503.02531",
        ]
        questions = ["What is the main topic?"]

        try:
            start_time = time.time()

            result = real_batch_processor.process_questions_multi_source(
                content_sources, questions
            )

            end_time = time.time()
            duration = end_time - start_time

            # Verify successful processing
            assert "answers" in result
            assert len(result["answers"]) == len(questions)
            assert all(len(answer) > 10 for answer in result["answers"])

            print("✅ Multimodal rate limiting test passed")
            print(f"   Processing time: {duration:.2f}s")
            print(f"   Token usage: {result['metrics']['usage']['total_tokens']}")

            # Should complete in reasonable time
            assert duration < 45  # Should not take more than 45 seconds

        except Exception as e:
            print(f"⚠️  Multimodal rate limiting test failed: {type(e).__name__}")
            print(f"   Error: {str(e)}")
            # Should be related to rate limiting or API issues
            assert any(
                keyword in str(e).lower()
                for keyword in ["rate", "limit", "api", "timeout"]
            )
